"""設定。環境変数 `EDGEOFF_*` または .env で上書きできる。"""

from pathlib import Path

import torch
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# 精度は fp32 統一（要件定義書 FR-2）。
# 本機のCPUは AVX512-BF16 を持たず、bf16行列積が fp32 の 1/620 しか出ないため。
DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDGEOFF_", env_file=".env", extra="ignore")

    hf_repo_id: str = "Qwen/Qwen2.5-3B-Instruct"

    # 精度（要件定義書 FR-2-2: 切替可能だが既定はfp32）
    dtype: str = "float32"

    # 生成設定（FR-4-5）
    prompt: str = "エッジデバイスで小規模言語モデルを動かす利点を3つ挙げてください。"
    max_new_tokens: int = 32

    # 計測設定
    cpu_threads: int = 8  # FR-3-5: 物理コア数に固定
    warmup_runs: int = 1  # 各反復ブロックの先頭に入れる捨て計測（基本設計6.3節）

    # スイープ設定
    offload_sweep_step: int = 4  # FR-4-1
    offload_sweep_repeats: int = 3  # FR-4-4
    boundary_sweep_repeats: int = 5  # FR-4-4
    noise_floor_repeats: int = 10  # FR-7-1

    results_dir: Path = Path("./results")

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return DTYPE_MAP[self.dtype]
        except KeyError as e:
            raise ValueError(
                f"未知のdtype: {self.dtype}（有効: {', '.join(DTYPE_MAP)}）"
            ) from e


settings = Settings()
