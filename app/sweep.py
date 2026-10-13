"""スイープ実行とCSV出力（FR-4、FR-5-1）。

計測順序の設計（基本設計6.3節・FR-4-3）:
    条件をまとめて反復する（AAAA→BBBB）のではなく、
    反復ブロックごとに全条件を回す（ABCD→ABCD→…）。
    前者だと、時間とともに変化する要因（サーマルスロットリング、
    OSのメモリ状態）が条件と交絡してしまうため。
"""

import csv
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from app.config import Settings
from app.generation import GenResult, generate
from app.model_host import ModelHost
from app.placement import Placement, boundary_sweep, build_contiguous, offload_sweep

FIELDS = [
    "sweep", "repeat", "label", "pattern", "n_layers", "n_gpu", "offload_ratio",
    "crossings_analytic", "crossings_per_forward", "crossing_bytes_per_forward",
    "ttft_ms", "prefill_ms", "tokenize_ms", "decode_mean_ms", "tokens_per_sec",
    "total_ms", "n_decode_steps", "peak_rss_mb", "peak_vram_mb",
    "prompt_tokens", "n_generated", "first_token_id", "token_ids",
]


@dataclass
class Row:
    sweep: str
    repeat: int
    placement: Placement
    result: GenResult

    def to_dict(self) -> dict:
        r, p = self.result, self.placement
        forwards = 1 + len(r.decode_times_s)
        return {
            "sweep": self.sweep,
            "repeat": self.repeat,
            "label": p.label,
            "pattern": p.pattern,
            "n_layers": p.n_layers,
            "n_gpu": p.n_gpu,
            "offload_ratio": round(p.offload_ratio, 4),
            "crossings_analytic": p.crossings,
            "crossings_per_forward": round(r.crossings / forwards, 4),
            "crossing_bytes_per_forward": round(r.crossing_bytes / forwards, 1),
            "ttft_ms": round(r.ttft_s * 1e3, 3),
            "prefill_ms": round(r.prefill_s * 1e3, 3),
            "tokenize_ms": round(r.tokenize_s * 1e3, 3),
            "decode_mean_ms": round(r.decode_mean_s * 1e3, 4),
            "tokens_per_sec": round(r.tokens_per_sec, 4),
            "total_ms": round(r.total_s * 1e3, 3),
            "n_decode_steps": len(r.decode_times_s),
            "peak_rss_mb": round(r.peak_rss_bytes / 2**20, 1),
            "peak_vram_mb": round(r.peak_vram_bytes / 2**20, 1),
            "prompt_tokens": r.prompt_tokens,
            "n_generated": len(r.token_ids),
            "first_token_id": r.token_ids[0] if r.token_ids else -1,
            "token_ids": " ".join(str(i) for i in r.token_ids),
        }


def _run_conditions(
    host: ModelHost,
    conditions: list[Placement],
    repeats: int,
    settings: Settings,
    sweep_name: str,
    verbose: bool = True,
) -> list[Row]:
    """条件を交互に反復して計測する（FR-4-3）。"""
    rows: list[Row] = []
    for rep in range(repeats):
        if verbose:
            print(f"--- {sweep_name} repeat {rep + 1}/{repeats} ---", flush=True)
        for p in conditions:
            host.apply_placement(p)
            # 反復ブロックごとの捨て計測（基本設計6.3節）
            if rep == 0:
                for _ in range(settings.warmup_runs):
                    generate(host, settings.prompt, min(4, settings.max_new_tokens))
            r = generate(host, settings.prompt, settings.max_new_tokens)
            rows.append(Row(sweep_name, rep, p, r))
            if verbose:
                print(
                    f"  {p.label:16s} n_gpu={p.n_gpu:2d} cross={p.crossings:2d} "
                    f"TTFT={r.ttft_s * 1e3:8.1f}ms decode={r.decode_mean_s * 1e3:8.2f}ms/tok "
                    f"({r.tokens_per_sec:6.2f} tok/s)",
                    flush=True,
                )
    return rows


def write_csv(rows: list[Row], path: Path, settings: Settings) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row.to_dict())
    # 再現性のため実行時の全設定を残す（NFR-2）
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings.model_dump(mode="json"),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "note": "PC環境（NVIDIA GPU + x86 CPU）の参考値。エッジNPUの性能予測には使用しない（NFR-3）",
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def run_offload_sweep(host: ModelHost, settings: Settings) -> list[Row]:
    """FR-4-1: オフロード率スイープ。"""
    conds = offload_sweep(host.n_layers, step=settings.offload_sweep_step)
    return _run_conditions(
        host, conds, settings.offload_sweep_repeats, settings, "offload"
    )


def run_boundary_sweep(host: ModelHost, settings: Settings) -> list[Row]:
    """FR-4-2: 境界数スイープ（オフロード率50%固定）。"""
    conds = boundary_sweep(host.n_layers)
    return _run_conditions(
        host, conds, settings.boundary_sweep_repeats, settings, "boundary"
    )


def run_noise_floor(host: ModelHost, settings: Settings) -> list[Row]:
    """FR-7-1: 同一条件を反復してばらつきを測る。

    条件は境界数スイープの基準（オフロード率50%の連続分割）に合わせる。
    """
    cond = [build_contiguous(host.n_layers, host.n_layers // 2)]
    return _run_conditions(
        host, cond, settings.noise_floor_repeats, settings, "noise_floor"
    )


def timestamped(results_dir: Path, name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{name}_{stamp}.csv"
