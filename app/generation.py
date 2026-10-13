"""prefill と decode を分離した greedy 生成ループ（基本設計6.1節）。

`model.generate()` を使わず自前でループを回すのは、prefill と decode_step を
別々に計測する必要があるため（FR-3-2）。サンプリングは greedy（argmax）のみ。
本検証の変数は層の配置だけであり、サンプリング方式は固定する。
"""

from dataclasses import dataclass, field

import psutil
import torch
from transformers import DynamicCache

from app.model_host import ModelHost
from app.stage_timer import StageTimer


@dataclass
class GenResult:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    tokenize_s: float = 0.0
    prefill_s: float = 0.0
    decode_times_s: list[float] = field(default_factory=list)
    peak_rss_bytes: int = 0
    peak_vram_bytes: int = 0
    crossings: int = 0
    crossing_bytes: int = 0
    prompt_tokens: int = 0

    @property
    def ttft_s(self) -> float:
        """最初のトークンが得られるまでの時間（tokenize + prefill）。"""
        return self.tokenize_s + self.prefill_s

    @property
    def decode_mean_s(self) -> float:
        d = self.decode_times_s
        return sum(d) / len(d) if d else 0.0

    @property
    def tokens_per_sec(self) -> float:
        m = self.decode_mean_s
        return 1.0 / m if m > 0 else 0.0

    @property
    def total_s(self) -> float:
        return self.ttft_s + sum(self.decode_times_s)


@torch.inference_mode()
def generate(
    host: ModelHost,
    prompt: str,
    max_new_tokens: int,
    timer: StageTimer | None = None,
) -> GenResult:
    """greedy生成を行い、ステージ別の所要時間を返す。"""
    timer = timer or StageTimer()
    model, tok = host.model, host.tokenizer
    dev = host.input_device()
    proc = psutil.Process()

    host.reset_peak_vram()
    host.router.stats.reset()
    result = GenResult()

    with timer.measure("tokenize"):
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to(dev)
    result.tokenize_s = timer.durations("tokenize")[-1]
    result.prompt_tokens = int(input_ids.shape[-1])

    cache = DynamicCache()

    # --- prefill: プロンプト全体を1回のforwardで処理する ---
    with timer.measure("prefill"):
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(dim=-1)
    result.prefill_s = timer.durations("prefill")[-1]

    result.token_ids.append(int(next_id.item()))
    eos_ids = _eos_ids(tok, model)

    # --- decode: 1トークンずつ ---
    for _ in range(max_new_tokens - 1):
        if result.token_ids[-1] in eos_ids:
            break
        with timer.measure("decode_step"):
            out = model(
                input_ids=next_id.view(1, 1), past_key_values=cache, use_cache=True
            )
            next_id = out.logits[:, -1, :].argmax(dim=-1)
        result.token_ids.append(int(next_id.item()))
        timer.record_memory_sample(proc.memory_info().rss)

    result.decode_times_s = timer.durations("decode_step")
    result.peak_rss_bytes = max(timer.peak_memory_bytes(), proc.memory_info().rss)
    result.peak_vram_bytes = host.peak_vram_bytes()
    result.crossings = host.router.stats.count
    result.crossing_bytes = host.router.stats.nbytes
    result.text = tok.decode(result.token_ids, skip_special_tokens=True)
    return result


def _eos_ids(tok, model) -> set[int]:
    ids: set[int] = set()
    for v in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    return ids
