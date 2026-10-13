"""llama.cpp の -ngl スイープと本実装（PyTorch）の曲線を突き合わせる（FR-8）。

FR-8-3 のとおり **絶対値は比較しない**。PyTorch側は fp32（重み12.33GB）、
llama.cpp側は Q4_K_M（同2.0GB）で読み出し量が約6倍違い、加えて
埋め込み層・出力層の扱いも異なる（本実装は常にGPU固定、llama.cppは
-ngl が層数を超えるまでCPU側）。比較するのは正規化した曲線の形状のみ。
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from app.analyze import _use_japanese_font, load_rows, summarize_offload


@dataclass
class Curve:
    name: str
    n_gpu: list[int]
    decode_ms: list[float]

    @property
    def tokens_per_sec(self) -> list[float]:
        return [1000.0 / d for d in self.decode_ms]

    @property
    def speedup(self) -> list[float]:
        base = self.decode_ms[0]
        return [base / d for d in self.decode_ms]

    @property
    def gain_fraction(self) -> list[float]:
        """全体の高速化のうち何%を得られたか（0〜100）。"""
        total = self.speedup[-1] - 1.0
        return [(s - 1.0) / total * 100 for s in self.speedup]

    def linear_fit(self) -> tuple[float, float, float]:
        """decode時間 vs GPU層数 の単回帰。(傾き, 切片, R²)"""
        xs, ys = self.n_gpu, self.decode_ms
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
        inter = my - slope * mx
        ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
        ss_tot = sum((y - my) ** 2 for y in ys)
        return slope, inter, 1 - ss_res / ss_tot

    def interpolate_at(self, n: int) -> float:
        """指定層数での decode時間を線形内挿する。"""
        for i in range(len(self.n_gpu) - 1):
            a, b = self.n_gpu[i], self.n_gpu[i + 1]
            if a <= n <= b:
                t = (n - a) / (b - a)
                return self.decode_ms[i] * (1 - t) + self.decode_ms[i + 1] * t
        return self.decode_ms[-1]


def load_pytorch(path: Path) -> Curve:
    rows = summarize_offload(path)
    return Curve("本実装 (PyTorch, fp32)",
                 [r["n_gpu"] for r in rows],
                 [r["decode_ms_mean"] for r in rows])


def load_llamacpp(path: Path) -> Curve:
    """llama-bench の CSV から decode（n_gen>0）の行だけを拾う。"""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pts = [
        (int(r["n_gpu_layers"]), 1000.0 / float(r["avg_ts"]))
        for r in rows
        if int(r["n_gen"]) > 0
    ]
    pts.sort()
    return Curve("llama.cpp (Vulkan, Q4_K_M)", [p[0] for p in pts], [p[1] for p in pts])


def report(pt: Curve, lc: Curve) -> str:
    out = []
    for c in (pt, lc):
        slope, inter, r2 = c.linear_fit()
        half = c.interpolate_at(c.n_gpu[-1] // 2)
        sp_half = c.decode_ms[0] / half
        frac = (sp_half - 1) / (c.speedup[-1] - 1) * 100
        out.append(
            f"[{c.name}]\n"
            f"  decode: {c.decode_ms[0]:7.2f} ms → {c.decode_ms[-1]:6.2f} ms "
            f"（全体 {c.speedup[-1]:5.2f}倍）\n"
            f"  層数への線形性: {slope:7.3f} ms/層  R²={r2:.4f}\n"
            f"  50%オフロード時: {sp_half:.2f}倍 = 全体の伸びの {frac:.1f}%"
        )
    return "\n\n".join(out)


def plot(pt: Curve, lc: Curve, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_japanese_font(plt)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    for c, style in ((pt, "o-"), (lc, "s--")):
        xs = [n / c.n_gpu[-1] * 100 for n in c.n_gpu]
        ax1.plot(xs, c.decode_ms, style, label=c.name)
        ax2.plot(xs, c.speedup, style, label=c.name)
        ax3.plot(xs, c.gain_fraction, style, label=c.name)

    ax1.set_yscale("log")
    ax1.set_ylabel("decode (ms / token, 対数軸)")
    ax1.set_title("① 絶対値は比較しない（量子化が違う）")

    ax2.set_ylabel("速度比（全CPU比）")
    ax2.set_title("② 正規化: どちらも終端で急峻")

    ax3.plot([0, 100], [0, 100], ":", color="gray", label="仮に線形なら")
    ax3.set_ylabel("得られた高速化 / 全体 (%)")
    ax3.set_title("③ 部分オフロードの効率（形状の一致）")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("オフロード率 (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("EdgeSLM_Offload Step 4: 実装をまたいだ曲線形状の突き合わせ（PC環境の参考値）")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def shape_correlation(pt: Curve, lc: Curve) -> float:
    """正規化した「得られた高速化の割合」の相関係数。

    両者の層数が違っても比較できるよう、オフロード率でそろえて内挿する。
    """
    ratios = [i / 10 for i in range(11)]
    def at(c: Curve, r: float) -> float:
        n = r * c.n_gpu[-1]
        d = c.interpolate_at(int(round(n)))
        sp = c.decode_ms[0] / d
        return (sp - 1) / (c.speedup[-1] - 1) * 100
    a = [at(pt, r) for r in ratios]
    b = [at(lc, r) for r in ratios]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


__all__ = ["Curve", "load_pytorch", "load_llamacpp", "report", "plot", "shape_correlation"]
