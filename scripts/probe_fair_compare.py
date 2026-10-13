"""PyTorch と llama.cpp を、より対等な条件で比べ直す。

Step 4 の比較は PyTorch=fp32(12.33GB) 対 llama.cpp=Q4_K_M(2.0GB) で、
読み出し量が6倍違う。絶対値の差の大半は「実装の差」ではなく「量子化の差」であり、
実装の優劣を論じる根拠にはならない（FR-8-3 で絶対値比較を禁じたのはこのため）。

fp32統一は「CPU側でbf16が使い物にならない」という制約から選んだものだが、
**全層GPU配置ならCPU側の層は存在しない**ため、この制約は効かない。
そこで全層GPU限定で bf16 でも測り、「1バイトあたりどれだけ速いか」で揃えて比べる。

**注意**: dtypeごとに必ず別プロセスで測ること。同一プロセスで2つのモデルを
続けて動かすと、先のモデルのメモリが解放しきれず後の計測が10倍以上遅くなる
（実測: 同一プロセスだと bf16 が 872ms、別プロセスなら 82.88ms）。
本スクリプトは子プロセスを起動してこれを回避している。

実行: .venv/Scripts/python.exe -u scripts/probe_fair_compare.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GPU_BANDWIDTH_GBPS = 791.8   # 要件定義書3.3節の実測値
LLAMACPP_MS = 6.14           # Step 4 実測（Q4_K_M、全層GPU）
LLAMACPP_GIB = 2007.4 / 1024

CHILD = '''
import sys, torch
sys.path.insert(0, r"{root}")
from app.config import settings
from app.generation import generate
from app.model_host import ModelHost
from app.placement import build_contiguous

torch.set_num_threads(settings.cpu_threads)
dt = torch.bfloat16 if sys.argv[1] == "bfloat16" else torch.float32
host = ModelHost(settings.hf_repo_id, dt)
host.load(build_contiguous(36, 36))          # 全層GPU
assert {{str(next(l.parameters()).device) for l in host.model.model.layers}} == {{"cuda:0"}}
generate(host, settings.prompt, 8)           # 捨て計測
vals = sorted(generate(host, settings.prompt, settings.max_new_tokens).decode_mean_s * 1e3
              for _ in range(3))
print(f"RESULT {{vals[1]:.4f}} {{torch.cuda.memory_allocated()/2**30:.4f}}")
'''


def measure(dtype: str) -> tuple[float, float]:
    """子プロセスで1つのdtypeだけを測る。"""
    script = ROOT / "scripts" / "_fair_child.py"
    script.write_text(CHILD.format(root=str(ROOT)), encoding="utf-8")
    out = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", "-u", str(script), dtype],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            _, ms, gib = line.split()
            return float(ms), float(gib)
    raise RuntimeError(f"計測失敗 ({dtype}):\n{out.stdout[-800:]}\n{out.stderr[-800:]}")


def main() -> int:
    print("全層GPU配置での比較（CPU側の層が無いので bf16 の制約は効かない）")
    print("dtypeごとに別プロセスで計測する\n")

    rows = []
    for dtype, label in (("float32", "PyTorch fp32"), ("bfloat16", "PyTorch bf16")):
        ms, gib = measure(dtype)
        rows.append((label, ms, gib))
        print(f"  {label:16s} decode={ms:8.2f} ms/token  重み={gib:5.2f} GiB", flush=True)
    rows.append(("llama.cpp Q4_K_M", LLAMACPP_MS, LLAMACPP_GIB))

    print(f"\n{'実装':18s} {'decode':>10s} {'重み':>9s} {'理論下限':>9s} {'理論比':>8s} {'実効帯域':>11s}")
    print("-" * 70)
    for label, ms, gib in rows:
        gb = gib * 2**30 / 1e9
        floor_ms = gb / GPU_BANDWIDTH_GBPS * 1e3
        print(f"{label:18s} {ms:9.2f}ms {gib:8.2f}GiB {floor_ms:8.2f}ms "
              f"{ms / floor_ms:7.2f}x {gb / (ms / 1e3):10.1f}GB/s")

    print("\n『理論比』が小さいほど帯域限界に近い＝1バイトあたりの効率が良い。")
    print("bf16はfp32の半分しか読まないのに遅い → decodeは帯域律速ではなく"
          "**層あたりの固定オーバーヘッド律速**であることが分かる。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
