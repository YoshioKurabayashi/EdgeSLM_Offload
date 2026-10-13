"""全層GPU配置が予測より140倍遅い問題の切り分け。

仮説を順に潰す:
  A. フック（_Router）が原因か        → フック無しの素の model.to('cuda') と比較
  B. VRAM逼迫によるアロケータ問題か   → メモリ状況を出す
  C. fp32のGEMMカーネルが遅いのか     → 単体GEMMを実測
  D. 初回のみ遅い（JIT等）か          → 複数回まわして推移を見る

実行: .venv/Scripts/python.exe -u scripts/diag_slow.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def sync():
    torch.cuda.synchronize()


def bench_forward(model, ids, cache_cls, label, iters=4):
    """prefill 1回 + decode を数回まわし、各回の時間を出す。"""
    from transformers import DynamicCache

    times = []
    with torch.inference_mode():
        cache = DynamicCache()
        sync(); t = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1)
        sync(); prefill = time.perf_counter() - t

        for _ in range(iters):
            sync(); t = time.perf_counter()
            out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(-1)
            sync(); times.append(time.perf_counter() - t)

    print(f"  [{label}] prefill={prefill*1e3:9.1f}ms  decode={[f'{x*1e3:.1f}' for x in times]}")
    return prefill, times


def main() -> int:
    torch.set_num_threads(settings.cpu_threads)
    print(f"torch={torch.__version__}  tf32_matmul={torch.backends.cuda.matmul.allow_tf32}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free={free/2**30:.2f} / total={total/2**30:.2f} GiB")

    # --- C: 単体GEMMの実測（モデルとは独立に、fp32カーネルの素の速度を見る） ---
    print("\n=== C. 単体GEMM (fp32, GPU) ===")
    for n in (2048, 4096):
        a = torch.randn(n, n, device="cuda"); b = torch.randn(n, n, device="cuda")
        for _ in range(3): a @ b
        sync(); t = time.perf_counter()
        for _ in range(20): a @ b
        sync(); dt = (time.perf_counter() - t) / 20
        print(f"  {n}^3: {dt*1e3:7.3f} ms  ({2*n**3/dt/1e12:6.2f} TFLOPS)")
        del a, b
    torch.cuda.empty_cache()

    # --- モデルをロード（CPU） ---
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    print("\n=== モデルロード ===")
    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(settings.hf_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_repo_id, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  load {time.perf_counter()-t:.1f}s")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": settings.prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids_cpu = enc["input_ids"]
    print(f"  prompt tokens = {ids_cpu.shape[-1]}")

    # --- A: フック無しの素のGPU実行 ---
    print("\n=== A. フック無し / 素の model.to('cuda') ===")
    t = time.perf_counter()
    model.to("cuda")
    sync()
    print(f"  to(cuda) {time.perf_counter()-t:.1f}s")
    alloc = torch.cuda.memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    print(f"  allocated={alloc:.2f} GiB  free={free/2**30:.2f} GiB")
    bench_forward(model, ids_cpu.to("cuda"), DynamicCache, "raw-gpu")

    # --- D: もう一度（初回のみ遅いのか） ---
    bench_forward(model, ids_cpu.to("cuda"), DynamicCache, "raw-gpu-2nd")

    # --- A': 同じモデルに本実装のフックを付けて比較 ---
    print("\n=== A'. 本実装のフックを装着（全層GPUのまま） ===")
    from app.model_host import ModelHost
    from app.placement import build_contiguous

    host = ModelHost(settings.hf_repo_id, torch.float32)
    host.model, host.tokenizer = model, tok
    from app.model_host import _DeviceSlot
    host._slots = [_DeviceSlot(device="cuda") for _ in model.model.layers]
    host._current = tuple("cuda" for _ in model.model.layers)
    host._install_hooks()
    bench_forward(model, ids_cpu.to("cuda"), DynamicCache, "hooked-gpu")

    p = build_contiguous(host.n_layers, host.n_layers)
    print(f"  crossings(analytic)={p.crossings} measured={host.router.stats.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
