"""CLIエントリポイント。

    uv run python -X utf8 -m app.cli check          環境と配置の確認
    uv run python -X utf8 -m app.cli offload        オフロード率スイープ（Step 1）
    uv run python -X utf8 -m app.cli noise          ノイズフロア確認（Step 2 / FR-7）
    uv run python -X utf8 -m app.cli boundary       境界数スイープ（Step 3）
    uv run python -X utf8 -m app.cli analyze <csv>  集計・プロット・回帰
"""

from pathlib import Path

import torch
import typer

from app.config import settings
from app.model_host import default_host

app = typer.Typer(add_completion=False, help="部分オフロード検証CLI")


def _host(initial=None):
    """initial を渡すと、ロード時点でその配置に置く（ピークRAMを下げる）。"""
    torch.set_num_threads(settings.cpu_threads)
    typer.echo(f"threads={torch.get_num_threads()} dtype={settings.dtype} model={settings.hf_repo_id}")
    h = default_host(settings.hf_repo_id, settings.torch_dtype)
    h.load(initial)
    typer.echo(f"layers={h.n_layers} edge={h.verify_edge_devices()}")
    return h


@app.command()
def check() -> None:
    """環境・配置・境界数の整合を確認する（計測はしない）。"""
    from app.placement import boundary_sweep, offload_sweep

    free, total = torch.cuda.mem_get_info()
    typer.echo(f"VRAM free={free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    h = _host()
    typer.echo("\n-- offload sweep 条件 --")
    for p in offload_sweep(h.n_layers, settings.offload_sweep_step):
        typer.echo(f"  {p.label:16s} ratio={p.offload_ratio:.2f} crossings={p.crossings:2d}")
    typer.echo("\n-- boundary sweep 条件 --")
    for p in boundary_sweep(h.n_layers):
        typer.echo(f"  {p.label:16s} ratio={p.offload_ratio:.2f} crossings={p.crossings:2d} {p.pattern}")


@app.command()
def offload() -> None:
    """FR-4-1: オフロード率スイープ。"""
    from app.sweep import run_offload_sweep, timestamped, write_csv

    h = _host()
    rows = run_offload_sweep(h, settings)
    out = write_csv(rows, timestamped(settings.results_dir, "offload"), settings)
    typer.echo(f"\n書き出し: {out}")


@app.command()
def noise() -> None:
    """FR-7: ノイズフロアの事前確認。"""
    from app.analyze import noise_floor
    from app.sweep import run_noise_floor, timestamped, write_csv

    from app.placement import build_contiguous
    h = _host(build_contiguous(36, 18))
    rows = run_noise_floor(h, settings)
    out = write_csv(rows, timestamped(settings.results_dir, "noise"), settings)
    nf = noise_floor(out)
    typer.echo(
        f"\nn={nf.n} mean={nf.mean_ms:.3f}ms sd={nf.sd_ms:.3f}ms "
        f"変動係数={nf.cv_percent:.3f}%"
    )
    # 基本設計5.3節: 予想信号は (36-2) x 45us = 1.53ms
    ok, msg = nf.verdict(expected_signal_ms=1.53)
    typer.echo(msg)
    typer.echo(f"書き出し: {out}")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def boundary() -> None:
    """FR-4-2: 境界数スイープ（中核実験）。"""
    from app.analyze import fit_boundary_cost, fit_boundary_cost_blocked
    from app.sweep import run_boundary_sweep, timestamped, write_csv

    from app.placement import build_blocked
    h = _host(build_blocked(36, 18))
    rows = run_boundary_sweep(h, settings)
    out = write_csv(rows, timestamped(settings.results_dir, "boundary"), settings)
    typer.echo("\n[単純な最小二乗（反復をブロックとして扱わない）]")
    typer.echo("  " + fit_boundary_cost(out).report())
    typer.echo("\n[対応のある回帰（反復をブロック因子として扱う）] ← 本命")
    typer.echo("  " + fit_boundary_cost_blocked(out).report())
    typer.echo(f"\n書き出し: {out}")


@app.command()
def analyze(csv_path: Path, plot: bool = True) -> None:
    """CSVを集計し、曲線をプロットする。"""
    from app.analyze import (
        fit_boundary_cost,
        fit_boundary_cost_blocked,
        plot_boundary_cost,
        plot_offload_curve,
        summarize_offload,
    )

    from app.analyze import load_rows

    # 境界数スイープは全条件が同じ n_gpu なので summarize_offload では1行に潰れる。
    # 判定は生データの「境界数が複数あり、オフロード率は単一」で行う。
    raw = load_rows(csv_path)
    is_boundary = (
        len({r["crossings_analytic"] for r in raw}) > 1
        and len({r["offload_ratio"] for r in raw}) == 1
    )

    if is_boundary:
        by_c: dict[int, list[float]] = {}
        label_of: dict[int, str] = {}
        for r in raw:
            c = int(r["crossings_analytic"])
            by_c.setdefault(c, []).append(float(r["decode_mean_ms"]))
            label_of[c] = r["label"]
        cs = sorted(by_c)
        base = sum(by_c[cs[0]]) / len(by_c[cs[0]])
        typer.echo(f"{'label':>12} {'境界数':>6} {'n':>4} {'decode_ms':>10} {'増分':>8}")
        for c in cs:
            m = sum(by_c[c]) / len(by_c[c])
            typer.echo(
                f"{label_of[c]:>12} {c:6d} {len(by_c[c]):4d} {m:10.2f} {m - base:+8.2f}"
            )
        typer.echo("\n[単純な最小二乗（反復をブロックとして扱わない）]")
        typer.echo("  " + fit_boundary_cost(csv_path).report())
        typer.echo("\n[対応のある回帰（反復をブロック因子として扱う）] ← 本命")
        typer.echo("  " + fit_boundary_cost_blocked(csv_path).report())
    else:
        rows = summarize_offload(csv_path)
        typer.echo(
            f"{'n_gpu':>6} {'ratio':>6} {'cross':>6} {'decode_ms':>10} {'tok/s':>8} {'比':>6}"
        )
        base_tps = rows[0]["tokens_per_sec_mean"] if rows else 1.0
        for r in rows:
            typer.echo(
                f"{r['n_gpu']:6d} {r['offload_ratio']:6.2f} {r['crossings']:6d} "
                f"{r['decode_ms_mean']:10.2f} {r['tokens_per_sec_mean']:8.2f} "
                f"{r['tokens_per_sec_mean'] / base_tps:6.2f}x"
            )

    if plot:
        fn = plot_boundary_cost if is_boundary else plot_offload_curve
        png = fn(csv_path, csv_path.with_suffix(".png"))
        typer.echo(f"\nプロット: {png}")


if __name__ == "__main__":
    app()
