"""結果の集計・プロット・回帰（FR-5-2、FR-5-3、FR-7）。

境界コストの推定は「境界数 vs decode時間」の最小二乗回帰で行う（基本設計2.1節）。
4水準はオフロード率が同一なので計算時間は理論上等しく、傾きが境界1回あたりの
コストになる。傾きが有意でない場合も、信頼区間の上限を「境界コストはこれ以下」
という定量的な結論として報告する。
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# 両側95%のt値。scipyを入れずに済ませるための小さな表（df=1..30）。
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T95.get(df, 1.960)  # df>30 は正規近似


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class NoiseFloor:
    n: int
    mean_ms: float
    sd_ms: float

    @property
    def cv_percent(self) -> float:
        return self.sd_ms / self.mean_ms * 100 if self.mean_ms else float("nan")

    def verdict(self, expected_signal_ms: float) -> tuple[bool, str]:
        """FR-7-2: 予想信号が2σを上回るか。"""
        two_sigma = 2 * self.sd_ms
        ok = expected_signal_ms > two_sigma
        msg = (
            f"予想信号 {expected_signal_ms:.3f}ms vs 2σ {two_sigma:.3f}ms → "
            + ("検出可能圏（本計測へ進んでよい）" if ok else
               "信号がノイズに埋もれる（反復増または生成長延長が必要）")
        )
        return ok, msg


def noise_floor(path: Path) -> NoiseFloor:
    """FR-7-1: 同一条件反復から decode時間の標準偏差を求める。"""
    vals = [float(r["decode_mean_ms"]) for r in load_rows(path)]
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return NoiseFloor(n=n, mean_ms=mean, sd_ms=sd)


@dataclass
class Regression:
    slope: float          # ms / 境界1回
    intercept: float
    se_slope: float
    df: int
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        h = t95(self.df) * self.se_slope
        return (self.slope - h, self.slope + h)

    @property
    def significant(self) -> bool:
        lo, hi = self.ci95
        return lo > 0 or hi < 0

    def report(self) -> str:
        lo, hi = self.ci95
        head = (
            f"境界1回あたり {self.slope * 1e3:.1f} µs "
            f"（95%CI: {lo * 1e3:.1f} 〜 {hi * 1e3:.1f} µs, n={self.n}）"
        )
        if self.significant:
            return head + "\n  → 傾きは有意。境界コストを検出した"
        return (
            head + f"\n  → 傾きは有意でない。**境界コストは {hi * 1e3:.1f} µs/回 以下**"
            "（これが本検証の結論となる上限値）"
        )


def fit_boundary_cost(path: Path) -> Regression:
    """FR-5-3: 境界数に対する decode時間の単回帰。"""
    rows = load_rows(path)
    xs = [float(r["crossings_analytic"]) for r in rows]
    ys = [float(r["decode_mean_ms"]) for r in rows]
    n = len(xs)
    if n < 3:
        raise ValueError(f"回帰には3点以上必要: {n}")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("境界数が全て同一で回帰できない")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    df = n - 2
    s2 = sum(r * r for r in resid) / df
    return Regression(
        slope=slope, intercept=intercept, se_slope=math.sqrt(s2 / sxx), df=df, n=n
    )


def fit_boundary_cost_blocked(path: Path) -> Regression:
    """反復をブロック因子として扱う対応のある回帰（within推定）。

    Step 1 の実測で、反復間に系統的なドリフト（反復2が反復1よりほぼ全点で速い）が
    あり、変動係数が5〜13%に達することが分かった。必要な分解能（約0.9%）に対して
    一桁大きく、`fit_boundary_cost()` の単純な最小二乗ではドリフトが誤差項に
    丸ごと入って傾きを検出できない。

    境界数スイープの4条件は各反復ブロック内で連続して測るため、ドリフトは
    4条件にほぼ等しく乗る（交互反復＝FR-4-3 の設計意図）。そこで各反復ごとに
    平均を引いてから回帰すると、共通のドリフトが相殺される。

    自由度は n - R - 1（R個の反復平均と傾きを推定するため）。
    """
    rows = load_rows(path)
    by_rep: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_rep[r["repeat"]].append(
            (float(r["crossings_analytic"]), float(r["decode_mean_ms"]))
        )

    n_reps = len(by_rep)
    xt: list[float] = []
    yt: list[float] = []
    for pairs in by_rep.values():
        if len(pairs) < 2:
            continue
        mx = sum(p[0] for p in pairs) / len(pairs)
        my = sum(p[1] for p in pairs) / len(pairs)
        for x, y in pairs:
            xt.append(x - mx)
            yt.append(y - my)

    n = len(xt)
    df = n - n_reps - 1
    if df < 1:
        raise ValueError(f"対応分析に必要なデータが不足（n={n}, 反復={n_reps}）")
    sxx = sum(x * x for x in xt)
    if sxx == 0:
        raise ValueError("境界数が全て同一で回帰できない")
    slope = sum(x * y for x, y in zip(xt, yt, strict=True)) / sxx
    ss_res = sum((y - slope * x) ** 2 for x, y in zip(xt, yt, strict=True))
    se = math.sqrt((ss_res / df) / sxx)
    # 切片は元スケールの全体平均から復元する（報告用）
    all_y = [float(r["decode_mean_ms"]) for r in rows]
    all_x = [float(r["crossings_analytic"]) for r in rows]
    intercept = sum(all_y) / len(all_y) - slope * (sum(all_x) / len(all_x))
    return Regression(slope=slope, intercept=intercept, se_slope=se, df=df, n=n)


def summarize_offload(path: Path) -> list[dict]:
    """オフロード率ごとに反復を集約する。"""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in load_rows(path):
        by_n[int(r["n_gpu"])].append(r)
    out = []
    for n_gpu in sorted(by_n):
        rs = by_n[n_gpu]
        tps = [float(r["tokens_per_sec"]) for r in rs]
        dec = [float(r["decode_mean_ms"]) for r in rs]
        ttft = [float(r["ttft_ms"]) for r in rs]
        out.append({
            "n_gpu": n_gpu,
            "offload_ratio": float(rs[0]["offload_ratio"]),
            "crossings": int(rs[0]["crossings_analytic"]),
            "repeats": len(rs),
            "decode_ms_mean": sum(dec) / len(dec),
            "tokens_per_sec_mean": sum(tps) / len(tps),
            "ttft_ms_mean": sum(ttft) / len(ttft),
            "peak_vram_mb": float(rs[0]["peak_vram_mb"]),
        })
    return out


def _use_japanese_font(plt) -> None:
    """日本語ラベルが豆腐にならないよう、あればCJKフォントを使う。

    matplotlib既定の DejaVu Sans は日本語グリフを持たない。
    Windows標準のゴシック体を優先し、無ければ既定のまま（英字は出る）。
    """
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # 負号が豆腐になるのを防ぐ
            return


def plot_offload_curve(path: Path, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_japanese_font(plt)
    rows = summarize_offload(path)
    xs = [r["offload_ratio"] * 100 for r in rows]
    ns = [r["n_gpu"] for r in rows]
    ys = [r["tokens_per_sec_mean"] for r in rows]
    ds = [r["decode_ms_mean"] for r in rows]
    base = ys[0] if ys else 1.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    # ① decode時間は層数に対して線形（加算モデルの検証）
    n = len(ns)
    mx, my = sum(ns) / n, sum(ds) / n
    sxx = sum((x - mx) ** 2 for x in ns)
    slope = sum((x - mx) * (y - my) for x, y in zip(ns, ds, strict=True)) / sxx
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(ns, ds, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ds)
    r2 = 1 - ss_res / ss_tot

    ax1.plot(ns, ds, "o", label="実測")
    ax1.plot(ns, [inter + slope * x for x in ns], "-", alpha=0.7,
             label=f"回帰: {slope:.2f} ms/層 (R²={r2:.4f})")
    ax1.set_xlabel("GPUに載せた層数")
    ax1.set_ylabel("decode (ms / token)")
    ax1.set_title("① decode時間は層数に線形")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ② その逆数なので tokens/sec は双曲線になる
    ax2.plot(xs, ys, "o-")
    ax2.set_xlabel("オフロード率 (%)")
    ax2.set_ylabel("tokens / sec")
    ax2.set_title("② 速度は双曲線（終端で急峻）")
    ax2.grid(alpha=0.3)

    # ③ 「半分載せても効果は薄い」を可視化
    gain = [(y / base - 1) / (ys[-1] / base - 1) * 100 for y in ys]
    ax3.plot(xs, gain, "o-")
    ax3.plot([0, 100], [0, 100], "--", color="gray", alpha=0.6, label="仮に線形なら")
    ax3.set_xlabel("オフロード率 (%)")
    ax3.set_ylabel("得られた高速化 / 全体 (%)")
    ax3.set_title("③ 部分オフロードの効率")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle("EdgeSLM_Offload: 部分オフロードの効果（PC環境の参考値・NPU予測には使用しない）")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def plot_boundary_cost(path: Path, out_png: Path) -> Path:
    """境界数スイープの結果と回帰直線（FR-5-2）。

    4条件はオフロード率が同一（GPU 18層 / CPU 18層）なので計算量は等しく、
    横軸方向の差はデバイス境界を跨ぐ回数だけに由来する。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_japanese_font(plt)
    rows = load_rows(path)
    by_c: dict[int, list[float]] = defaultdict(list)
    label_of: dict[int, str] = {}
    for r in rows:
        c = int(r["crossings_analytic"])
        by_c[c].append(float(r["decode_mean_ms"]))
        label_of[c] = r["label"]

    cs = sorted(by_c)
    means = [sum(by_c[c]) / len(by_c[c]) for c in cs]
    sems = [
        (math.sqrt(sum((v - m) ** 2 for v in by_c[c]) / (len(by_c[c]) - 1))
         / math.sqrt(len(by_c[c])))
        for c, m in zip(cs, means, strict=True)
    ]
    reg = fit_boundary_cost_blocked(path)
    lo, hi = reg.ci95

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.errorbar(cs, means, yerr=sems, fmt="o", capsize=4, label="実測（平均±標準誤差）")
    xr = [min(cs), max(cs)]
    ax1.plot(xr, [reg.intercept + reg.slope * x for x in xr], "-", alpha=0.8,
             label=f"回帰: {reg.slope * 1e3:.0f} µs/境界")
    ax1.fill_between(xr,
                     [reg.intercept + lo * x for x in xr],
                     [reg.intercept + hi * x for x in xr],
                     alpha=0.15, label="95%信頼区間")
    ax1.set_xlabel("デバイス境界を跨ぐ回数 / token")
    ax1.set_ylabel("decode (ms / token)")
    ax1.set_title("境界数と decode 時間（オフロード率は50%固定）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    base = means[0]
    ax2.bar([label_of[c] for c in cs], [m - base for m in means],
            yerr=sems, capsize=4, color="tab:orange", alpha=0.85)
    ax2.set_ylabel(f"境界{cs[0]}回の条件からの増分 (ms)")
    ax2.set_xlabel("配置パターン")
    ax2.set_title("同じ18層GPUでも、分断すると遅くなる")
    for i, c in enumerate(cs):
        ax2.annotate(f"境界{c}回", (i, 0), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("EdgeSLM_Offload: デバイス境界のコスト（PC環境の参考値）")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
