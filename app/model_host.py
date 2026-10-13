"""モデルの保持と層配置の変更（基本設計4節・7.1節）。

設計の要点:

1. **再ロードしない**。条件を変えるたびにディスクから読み直すのではなく、
   ロード済みモデルの層だけをデバイス間で移動する（基本設計7.1節）。

2. **デバイス境界の転送は自前のフックで行う**。`accelerate.dispatch_model()` でも
   同等のことはできるが、本検証は境界を跨ぐ回数とバイト数そのものが測定対象
   （FR-3-4）であるため、転送を自分で握って数える方式を採る。

3. **補助テンソルは1回だけ転送する**。RoPEのcos/sin等は全層が同じテンソルを
   共有して受け取るため、素朴に実装すると層ごとに転送してしまう。境界コストは
   バイト数ではなく転送1回あたりのレイテンシが支配的（要件定義書3.3節）なので、
   これを放置すると測定対象の信号が水増しされる。デバイスごとにメモ化して
   1回に抑える。境界のペイロードとして数えるのは hidden_states のみ。
"""

import gc
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.placement import CUDA, EDGE_DEVICE, Placement


@dataclass
class _DeviceSlot:
    """層の現在の配置先。フックはこのオブジェクトを見るので、
    配置変更時にフックを付け替える必要がない。"""

    device: str


@dataclass
class CrossingStats:
    """境界跨ぎの実測値（FR-3-4）。"""

    count: int = 0
    nbytes: int = 0

    def reset(self) -> None:
        self.count = 0
        self.nbytes = 0


@dataclass
class _Router:
    """フック本体。層の入力を、その層が置かれたデバイスへ移送する。"""

    stats: CrossingStats = field(default_factory=CrossingStats)
    # (id(元テンソル), 転送先) -> (元テンソル, 転送後テンソル)
    # 元テンソルへの参照を保持するのは、GCによるid再利用で誤ヒットするのを防ぐため。
    _memo: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def reset_forward(self) -> None:
        """forward 1回ごとにメモを破棄する（テンソルが作り直されるため）。"""
        self._memo.clear()

    def _move_aux(self, t: torch.Tensor, device: str) -> torch.Tensor:
        """補助テンソルの移送。同一テンソル・同一転送先なら再利用する。"""
        if t.device.type == device:
            return t
        key = (id(t), device)
        hit = self._memo.get(key)
        if hit is not None:
            return hit[1]
        moved = t.to(device)
        self._memo[key] = (t, moved)
        return moved

    def _walk(self, obj, device: str):
        """args/kwargs を再帰的に辿り、テンソルだけを移送する。

        Cache オブジェクトやint等はそのまま返す（Cacheは層ごとに
        自分のデバイス上へ書き込まれるため移送不要）。
        """
        if isinstance(obj, torch.Tensor):
            return self._move_aux(obj, device)
        if isinstance(obj, tuple):
            return tuple(self._walk(o, device) for o in obj)
        if isinstance(obj, list):
            return [self._walk(o, device) for o in obj]
        return obj

    def route(self, args: tuple, kwargs: dict, device: str) -> tuple[tuple, dict]:
        # hidden_states（第1引数、またはキーワード）が境界のペイロード。
        # 層ごとに新しく作られるテンソルなのでメモ化せず、毎回数える。
        new_args = list(args)
        hs = None
        if args and isinstance(args[0], torch.Tensor):
            hs, idx, key = args[0], 0, None
        elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
            hs, idx, key = kwargs["hidden_states"], None, "hidden_states"

        if hs is not None and hs.device.type != device:
            self.stats.count += 1
            self.stats.nbytes += hs.numel() * hs.element_size()
            moved = hs.to(device)
            if key is None:
                new_args[idx] = moved
            else:
                kwargs = {**kwargs, "hidden_states": moved}

        # 残りの引数（attention_mask / position_embeddings / cache_position 等）
        start = 1 if (hs is not None and key is None) else 0
        for i in range(start, len(new_args)):
            new_args[i] = self._walk(new_args[i], device)
        new_kwargs = {
            k: (v if k == "hidden_states" else self._walk(v, device))
            for k, v in kwargs.items()
        }
        return tuple(new_args), new_kwargs


class ModelHost:
    """モデルのロード・配置変更・トークナイザ保持をまとめる。"""

    def __init__(self, repo_id: str, dtype: torch.dtype) -> None:
        self.repo_id = repo_id
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.router = _Router()
        self._slots: list[_DeviceSlot] = []
        self._current: tuple[str, ...] | None = None
        self._handles: list = []

    # ------------------------------------------------------------------ load

    def load(self, initial: Placement | None = None) -> None:
        """CPU上にロードし、埋め込み層・出力層・normをGPUへ固定する。

        `initial` を渡すと、fp32化と同時にその配置先へ直接送る。CPU側に36層分の
        fp32(12.33GB)を一度も作らずに済むため、ピークRAMが大きく下がる。
        全条件が50%オフロードの境界数スイープでは特に効く。

        embed_tokens と lm_head は tie_word_embeddings=True により同一テンソル
        なので、片方を移すともう片方も移る（基本設計4.2節）。

        **ロード時のピークRAMを抑える工夫**: チェックポイントはbf16なので、
        `from_pretrained(dtype=float32)` と指定すると変換の過程でbf16(5.8GB)と
        fp32(12.3GB)が同時に存在し、ピークが18GB規模になる。実測でこれが原因の
        スワップが発生したため、**bf16でロードしてから層単位でfp32へ変換する**。
        bf16はfp32の上位16bitなので変換に情報の損失はなく、結果は
        `dtype=float32` で直接ロードした場合とビット単位で同一になる。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        # まずbf16でロード（fp32の約半分のメモリで済む）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.model.eval()

        inner = self.model.model
        # 埋め込み・出力・norm はGPUへ移しつつfp32化する（CPU側のbf16は解放される）
        inner.embed_tokens.to(device=EDGE_DEVICE, dtype=self.dtype)
        inner.norm.to(device=EDGE_DEVICE, dtype=self.dtype)
        if hasattr(inner, "rotary_emb"):
            inner.rotary_emb.to(device=EDGE_DEVICE, dtype=self.dtype)
        self.model.lm_head.to(device=EDGE_DEVICE, dtype=self.dtype)

        # decoder層は1層ずつfp32化する。bf16が減りながらfp32が増えるので、
        # 両方が丸ごと同時に存在するピークを避けられる。
        # initial が指定されていれば、その場で配置先へも送る（CPU側にfp32を溜めない）。
        if initial is not None and initial.n_layers != len(inner.layers):
            raise ValueError(
                f"initialの層数({initial.n_layers})がモデル({len(inner.layers)})と不一致"
            )
        devices = initial.layers if initial is not None else ("cpu",) * len(inner.layers)
        for layer, dev in zip(inner.layers, devices, strict=True):
            layer.to(device=dev, dtype=self.dtype)
        gc.collect()

        self._slots = [_DeviceSlot(device=d) for d in devices]
        self._current = tuple(devices)
        self._install_hooks()

    @property
    def n_layers(self) -> int:
        return len(self.model.model.layers)

    # ----------------------------------------------------------------- hooks

    def _install_hooks(self) -> None:
        inner = self.model.model

        # forward の開始時にメモを破棄する
        def _reset(_module, _args, _kwargs):
            self.router.reset_forward()
            return None

        self._handles.append(
            self.model.register_forward_pre_hook(_reset, with_kwargs=True)
        )

        for layer, slot in zip(inner.layers, self._slots, strict=True):
            def make(slot=slot):
                def hook(_module, args, kwargs):
                    return self.router.route(args, kwargs, slot.device)

                return hook

            self._handles.append(
                layer.register_forward_pre_hook(make(), with_kwargs=True)
            )

        # 最終層 → norm(GPU固定) の境界もここで数える
        def norm_hook(_module, args, kwargs):
            return self.router.route(args, kwargs, EDGE_DEVICE)

        self._handles.append(
            inner.norm.register_forward_pre_hook(norm_hook, with_kwargs=True)
        )

    # ------------------------------------------------------------- placement

    def apply_placement(self, placement: Placement) -> int:
        """層を指定デバイスへ移動する。既に正しい位置にある層は動かさない。

        Returns: 実際に移動した層数
        """
        if placement.n_layers != self.n_layers:
            raise ValueError(
                f"配置の層数({placement.n_layers})がモデル({self.n_layers})と不一致"
            )
        moved = 0
        for layer, slot, target, current in zip(
            self.model.model.layers, self._slots, placement.layers,
            self._current, strict=True,
        ):
            if current != target:
                layer.to(target)
                moved += 1
            slot.device = target
        self._current = placement.layers
        return moved

    def input_device(self) -> str:
        """入力トークンを置くべきデバイス（埋め込み層と同じ）。"""
        return EDGE_DEVICE

    # ---------------------------------------------------------------- memory

    def reset_peak_vram(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_vram_bytes(self) -> int:
        return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    def verify_edge_devices(self) -> dict[str, str]:
        """埋め込み層・出力層が想定どおりGPU上にあるかの確認用。"""
        inner = self.model.model
        return {
            "embed_tokens": str(next(inner.embed_tokens.parameters()).device),
            "lm_head": str(next(self.model.lm_head.parameters()).device),
            "norm": str(next(inner.norm.parameters()).device),
            "tied": str(
                self.model.lm_head.weight is inner.embed_tokens.weight
            ),
        }


def default_host(repo_id: str, dtype: torch.dtype = torch.float32) -> ModelHost:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できない環境では本検証は実施できない")
    return ModelHost(repo_id=repo_id, dtype=dtype)


__all__ = ["ModelHost", "CrossingStats", "default_host", "CUDA"]
