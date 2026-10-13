"""ステージ別計測ユーティリティ。

EdgeSLM（`EdgeSLM/app/profiling/stage_timer.py`）からコピーし、本プロジェクト向けに
**CUDA同期を追加**した（基本設計6.2節）。参照ではなくコピーとすることで、
EdgeSLM本体への依存を持たない（要件定義書 NFR-1）。

追加点: CUDAの演算はカーネル投入後に非同期で進むため、計測区間の前後で
`torch.cuda.synchronize()` を呼ばないと投入時刻しか測れない。
"""

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def sync() -> None:
    """GPUの処理完了を待つ。CUDAが無い環境では何もしない。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, cuda_sync: bool = True) -> None:
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._peak_memory_bytes = 0
        self._cuda_sync = cuda_sync

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """`stage`の所要時間を計測する。ブロック内で例外が発生した場合は記録しない。"""
        if self._cuda_sync:
            sync()
        start = time.perf_counter()
        yield
        if self._cuda_sync:
            sync()
        self._durations[stage].append(time.perf_counter() - start)

    def durations(self, stage: str) -> list[float]:
        return list(self._durations.get(stage, []))

    def total_seconds(self, stage: str) -> float:
        return sum(self._durations.get(stage, []))

    def call_count(self, stage: str) -> int:
        return len(self._durations.get(stage, []))

    def mean_seconds(self, stage: str) -> float:
        d = self._durations.get(stage, [])
        return sum(d) / len(d) if d else 0.0

    def record_memory_sample(self, rss_bytes: int) -> None:
        self._peak_memory_bytes = max(self._peak_memory_bytes, rss_bytes)

    def peak_memory_bytes(self) -> int:
        return self._peak_memory_bytes

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for stage, durations in self._durations.items():
            result[stage] = {
                "total_seconds": sum(durations),
                "count": len(durations),
                "mean_seconds": sum(durations) / len(durations) if durations else 0.0,
            }
        return result
