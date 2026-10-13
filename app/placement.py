"""層→デバイスの割り当てマップを生成し、境界数を解析的に算出する。

基本設計 4.3節・4.4節に対応。

配置は `list[str]`（長さ = 層数、各要素が "cpu" または "cuda"）で表現する。
埋め込み層・出力層はGPU固定のため配置リストには含めず、境界数の算出時にのみ
系列の両端として考慮する（基本設計4.2節）。
"""

from dataclasses import dataclass
from typing import Literal

CPU = "cpu"
CUDA = "cuda"

Policy = Literal["contiguous", "blocked", "explicit"]

# 埋め込み層・出力層の固定デバイス（基本設計4.2節）。
# tie_word_embeddings=True のため embed_tokens と lm_head は同一テンソルであり、
# 別々のデバイスには置けない。
EDGE_DEVICE = CUDA


@dataclass(frozen=True)
class Placement:
    """1条件分の層配置。

    layers: 各層のデバイス（長さ = 層数）
    label:  結果CSVに記録する条件名
    """

    layers: tuple[str, ...]
    label: str

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_gpu(self) -> int:
        return sum(1 for d in self.layers if d == CUDA)

    @property
    def offload_ratio(self) -> float:
        return self.n_gpu / self.n_layers if self.n_layers else 0.0

    @property
    def crossings(self) -> int:
        """デバイスが変わる箇所の数（基本設計4.4節）。

        embed_tokens(GPU) → layer0 → ... → layer_last → norm/lm_head(GPU)
        という系列で、隣接する要素のデバイスが異なる箇所を数える。
        """
        seq = (EDGE_DEVICE, *self.layers, EDGE_DEVICE)
        return sum(1 for a, b in zip(seq, seq[1:], strict=False) if a != b)

    @property
    def pattern(self) -> str:
        """"GGGGCCCC" 形式の可読表現。ログとCSVの目視確認用。"""
        return "".join("G" if d == CUDA else "C" for d in self.layers)


def build_contiguous(n_layers: int, n_gpu: int) -> Placement:
    """先頭 n_gpu 層をGPU、残りをCPUに置く（連続分割）。"""
    if not 0 <= n_gpu <= n_layers:
        raise ValueError(f"n_gpu は 0〜{n_layers} の範囲: {n_gpu}")
    layers = tuple(CUDA if i < n_gpu else CPU for i in range(n_layers))
    return Placement(layers=layers, label=f"contiguous_n{n_gpu}")


def build_blocked(n_layers: int, block_size: int, start: str = CUDA) -> Placement:
    """block_size 層ずつGPU/CPUを交互に配置する（ブロック交互分割）。

    オフロード率を50%に保ったまま境界数だけを振るために使う（基本設計2.1節）。
    n_layers が block_size*2 の倍数でない場合、末尾は途中で打ち切られるため
    オフロード率が50%からずれる点に注意（Qwen2.5-3Bの36層では
    block_size=1,2,6,18 のいずれもちょうど50%になる）。
    """
    if block_size < 1:
        raise ValueError(f"block_size は1以上: {block_size}")
    other = CPU if start == CUDA else CUDA
    layers = tuple(
        start if (i // block_size) % 2 == 0 else other for i in range(n_layers)
    )
    return Placement(layers=layers, label=f"blocked_b{block_size}")


def build_explicit(layers: list[str], label: str = "explicit") -> Placement:
    """明示リストをそのまま使う。"""
    for d in layers:
        if d not in (CPU, CUDA):
            raise ValueError(f"デバイスは '{CPU}' か '{CUDA}': {d}")
    return Placement(layers=tuple(layers), label=label)


def build_placement(policy: Policy, n_layers: int, **kwargs) -> Placement:
    """ポリシー名から配置を生成するディスパッチャ。"""
    if policy == "contiguous":
        return build_contiguous(n_layers, kwargs["n_gpu"])
    if policy == "blocked":
        return build_blocked(n_layers, kwargs["block_size"], kwargs.get("start", CUDA))
    if policy == "explicit":
        return build_explicit(kwargs["layers"], kwargs.get("label", "explicit"))
    raise ValueError(f"未知のポリシー: {policy}")


def offload_sweep(n_layers: int, step: int = 4) -> list[Placement]:
    """オフロード率スイープの条件列（FR-4-1）。

    0, step, 2*step, ... と振り、最後は必ず n_layers（全層GPU）を含める。
    """
    points = list(range(0, n_layers + 1, step))
    if points[-1] != n_layers:
        points.append(n_layers)
    return [build_contiguous(n_layers, n) for n in points]


def boundary_sweep(n_layers: int, block_sizes: tuple[int, ...] = (18, 6, 2, 1)) -> list[Placement]:
    """境界数スイープの条件列（FR-4-2、基本設計2.1節のL1〜L4）。

    全条件でオフロード率50%を保ち、境界数のみが変わる。
    """
    placements = [build_blocked(n_layers, b) for b in block_sizes]
    ratios = {p.offload_ratio for p in placements}
    if len(ratios) != 1:
        raise ValueError(
            f"境界数スイープはオフロード率が同一である必要がある: {sorted(ratios)}"
        )
    return placements
