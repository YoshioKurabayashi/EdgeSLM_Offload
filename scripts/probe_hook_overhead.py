"""計測装置（フック）自身のオーバーヘッドを測る。

全層GPU配置の実測(69.89ms/token)が基本設計の予測(15.6ms/token)より4.5倍遅い。
帯域律速の予測が外れる要因として、(a)自前フックのPythonコスト、
(b)batch=1のGEMVが帯域を使い切れないこと、(c)transformers自体のPythonコスト、
が考えられる。(a)は計測装置由来の人工物なので、大きいなら削る必要がある
（ベースラインが膨らむと境界コストの信号が相対的に埋もれるため）。

同一プロセス・同一モデルで、フックの有無だけを変えて比較する。

実行: .venv/Scripts/python.exe -u scripts/probe_hook_overhead.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.model_host import default_host  # noqa: E402
from app.placement import build_contiguous  # noqa: E402


def bench(model, tok, prompt, steps=12):
    from transformers import DynamicCache

    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"].to("cuda")
    times = []
    with torch.inference_mode():
        cache = DynamicCache()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1)
        for _ in range(steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(-1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    times = sorted(times)[2:-2]  # 外れ値を落とす
    return sum(times) / len(times) * 1e3


def main() -> int:
    torch.set_num_threads(settings.cpu_threads)
    host = default_host(settings.hf_repo_id, settings.torch_dtype)
    host.load()
    host.apply_placement(build_contiguous(host.n_layers, host.n_layers))  # 全層GPU
    print(f"全層GPU配置  VRAM={torch.cuda.memory_allocated()/2**30:.2f} GiB\n")

    with_hooks = bench(host.model, host.tokenizer, settings.prompt)
    print(f"  フック有り : {with_hooks:7.2f} ms/token")

    for h in host._handles:
        h.remove()
    host._handles.clear()
    without = bench(host.model, host.tokenizer, settings.prompt)
    print(f"  フック無し : {without:7.2f} ms/token")

    diff = with_hooks - without
    print(f"\n  フックのコスト = {diff:6.2f} ms/token "
          f"({diff / with_hooks * 100:.1f}% / 1層あたり {diff / host.n_layers * 1e3:.0f} us)")
    print(f"  フック無しでも予測(15.6ms)の {without / 15.6:.1f} 倍 "
          "→ 残りは transformers 本体と batch=1 GEMV の非効率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
