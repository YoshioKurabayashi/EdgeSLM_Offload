
import sys, torch
sys.path.insert(0, r"D:\AI\source\n_chap5\EdgeSLM_Offload")
from app.config import settings
from app.generation import generate
from app.model_host import ModelHost
from app.placement import build_contiguous

torch.set_num_threads(settings.cpu_threads)
dt = torch.bfloat16 if sys.argv[1] == "bfloat16" else torch.float32
host = ModelHost(settings.hf_repo_id, dt)
host.load(build_contiguous(36, 36))          # 全層GPU
assert {str(next(l.parameters()).device) for l in host.model.model.layers} == {"cuda:0"}
generate(host, settings.prompt, 8)           # 捨て計測
vals = sorted(generate(host, settings.prompt, settings.max_new_tokens).decode_mean_s * 1e3
              for _ in range(3))
print(f"RESULT {vals[1]:.4f} {torch.cuda.memory_allocated()/2**30:.4f}")
