"""Step 1 実装のスモークテスト。

実モデルをロードし、以下を確認する:
  1. 埋め込み層・出力層がGPU固定で、重み共有(tied)が成立していること
  2. 境界数の実測値が placement.py の解析値と一致すること（FR-3-4）
  3. 全CPU配置と全GPU配置で生成トークン列が一致すること（FR-6-1）
  4. 速度が基本設計5.1節の予測とおおむね整合すること

実行: .venv/Scripts/python.exe -u scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.generation import generate  # noqa: E402
from app.model_host import default_host  # noqa: E402
from app.placement import build_blocked, build_contiguous  # noqa: E402

MAX_NEW = 8  # スモークテストなので短く


def main() -> int:
    torch.set_num_threads(settings.cpu_threads)
    print(f"threads={torch.get_num_threads()} dtype={settings.dtype}")

    t0 = time.perf_counter()
    host = default_host(settings.hf_repo_id, settings.torch_dtype)
    host.load()
    print(f"[load] {time.perf_counter() - t0:.1f}s  layers={host.n_layers}")

    print("[edge devices]", host.verify_edge_devices())
    assert host.verify_edge_devices()["tied"] == "True", "重み共有が成立していない"

    n = host.n_layers
    cases = [
        build_contiguous(n, n),      # 全GPU
        build_contiguous(n, 0),      # 全CPU
        build_contiguous(n, n // 2),  # 半分
        build_blocked(n, 1),          # 1層ごと交互（境界最多）
    ]

    baseline_ids = None
    ok = True
    for p in cases:
        moved = host.apply_placement(p)
        r = generate(host, settings.prompt, MAX_NEW)

        analytic = p.crossings
        # 境界数の実測は「1トークンあたり」に直す（prefill 1回 + decode N回）
        forwards = 1 + len(r.decode_times_s)
        measured = r.crossings / forwards

        status = "OK " if abs(measured - analytic) < 1e-9 else "NG "
        if status == "NG ":
            ok = False
        print(
            f"[{p.label:16s}] pattern={p.pattern}\n"
            f"    moved={moved:2d} ratio={p.offload_ratio:.2f} "
            f"crossings: analytic={analytic:2d} measured={measured:.2f} {status}\n"
            f"    TTFT={r.ttft_s * 1e3:8.1f}ms  decode={r.decode_mean_s * 1e3:7.2f}ms/tok "
            f"({r.tokens_per_sec:6.2f} tok/s)  VRAM={r.peak_vram_bytes / 2**30:.2f}GiB"
        )

        if baseline_ids is None:
            baseline_ids = r.token_ids
            print(f"    text={r.text!r}")
        elif r.token_ids != baseline_ids:
            first = next(
                (i for i, (a, b) in enumerate(zip(baseline_ids, r.token_ids)) if a != b),
                min(len(baseline_ids), len(r.token_ids)),
            )
            print(f"    !! トークン列が全GPU配置と分岐: {first}トークン目から")
            ok = False
        else:
            print("    トークン列は全GPU配置と一致")

    print("\n=== SMOKE TEST:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
