"""ロードと配置変更のピークRAM/VRAMを測る。

実測でロード時のピークRAMがスワップを誘発したため、その改善を確認する。
生成はしないので短時間で終わる。

実行: .venv/Scripts/python.exe -u scripts/probe_memory.py
"""

import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.model_host import default_host  # noqa: E402
from app.placement import build_blocked, build_contiguous  # noqa: E402

proc = psutil.Process()


def snap(tag: str) -> None:
    rss = proc.memory_info().rss / 2**30
    vram = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    free = psutil.virtual_memory().available / 2**30
    print(f"  {tag:34s} RSS={rss:6.2f} GiB  VRAM={vram:6.2f} GiB  空きRAM={free:6.2f} GiB",
          flush=True)


def main() -> int:
    torch.set_num_threads(settings.cpu_threads)
    print(f"model={settings.hf_repo_id} dtype={settings.dtype}")
    snap("起動直後")

    t = time.perf_counter()
    host = default_host(settings.hf_repo_id, settings.torch_dtype)
    host.load()
    print(f"[load] {time.perf_counter() - t:.1f}s")
    snap("ロード後（全層CPU）")

    n = host.n_layers
    for p in (build_contiguous(n, n), build_contiguous(n, 0),
              build_contiguous(n, n // 2), build_blocked(n, 1)):
        t = time.perf_counter()
        moved = host.apply_placement(p)
        dt = time.perf_counter() - t
        snap(f"{p.label} (moved={moved}, {dt:.1f}s)")

    print("\nピークRSS(プロセス全体):"
          f" {proc.memory_info().peak_wset / 2**30:.2f} GiB"
          if hasattr(proc.memory_info(), "peak_wset") else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
