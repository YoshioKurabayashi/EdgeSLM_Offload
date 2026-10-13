"""analyze.py の単体テスト。

境界コストの回帰（FR-5-3）とノイズフロア判定（FR-7-2）は本検証の結論を
左右するため、既知の入力から既知の答えが出ることを固定しておく。
"""

import csv

import pytest

from app.analyze import (
    fit_boundary_cost,
    noise_floor,
    summarize_offload,
    t95,
)


def write_csv(path, rows):
    fields = [
        "sweep", "repeat", "label", "pattern", "n_layers", "n_gpu", "offload_ratio",
        "crossings_analytic", "crossings_per_forward", "crossing_bytes_per_forward",
        "ttft_ms", "prefill_ms", "tokenize_ms", "decode_mean_ms", "tokens_per_sec",
        "total_ms", "n_decode_steps", "peak_rss_mb", "peak_vram_mb",
        "prompt_tokens", "n_generated", "first_token_id", "token_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            base = dict.fromkeys(fields, 0)
            base.update(r)
            w.writerow(base)
    return path


def boundary_row(crossings, decode_ms, repeat=0):
    return {
        "sweep": "boundary", "repeat": repeat, "label": f"b{crossings}",
        "n_layers": 36, "n_gpu": 18, "offload_ratio": 0.5,
        "crossings_analytic": crossings, "decode_mean_ms": decode_ms,
        "tokens_per_sec": 1000.0 / decode_ms, "ttft_ms": 700.0, "peak_vram_mb": 6500,
    }


class TestRegression:
    def test_recovers_known_slope(self, tmp_path):
        """decode = 200ms + 0.07ms × 境界数 という合成データから傾きを復元できるか。

        0.07ms/境界 = 70µs は、スモークテストで観測された予備値に相当する。
        """
        rows = [
            boundary_row(c, 200.0 + 0.07 * c, rep)
            for rep in range(5)
            for c in (2, 6, 18, 36)
        ]
        p = write_csv(tmp_path / "b.csv", rows)
        reg = fit_boundary_cost(p)
        assert reg.slope == pytest.approx(0.07, abs=1e-6)
        assert reg.intercept == pytest.approx(200.0, abs=1e-6)
        assert reg.significant, "ノイズ皆無なら有意になるはず"
        assert "境界1回あたり 70.0" in reg.report()

    def test_flat_data_is_not_significant(self, tmp_path):
        """境界数を変えても時間が変わらない（＝境界コストが無い）場合。

        このとき「有意でない」と判定し、上限値を結論として示すこと（FR-5-3）。
        """
        decodes = [200.0, 200.5, 199.5, 200.2]  # 境界数と無関係なばらつき
        rows = [
            boundary_row(c, decodes[i % 4], rep)
            for rep in range(5)
            for i, c in enumerate((2, 6, 18, 36))
        ]
        p = write_csv(tmp_path / "flat.csv", rows)
        reg = fit_boundary_cost(p)
        assert not reg.significant
        assert "以下" in reg.report(), "上限値を結論として示すこと"

    def test_needs_at_least_three_points(self, tmp_path):
        p = write_csv(tmp_path / "few.csv", [boundary_row(2, 200.0)])
        with pytest.raises(ValueError, match="3点以上"):
            fit_boundary_cost(p)

    def test_rejects_constant_crossings(self, tmp_path):
        rows = [boundary_row(2, 200.0 + i) for i in range(4)]
        p = write_csv(tmp_path / "const.csv", rows)
        with pytest.raises(ValueError, match="回帰できない"):
            fit_boundary_cost(p)


class TestNoiseFloor:
    def test_statistics(self, tmp_path):
        vals = [100.0, 102.0, 98.0, 101.0, 99.0]
        rows = [boundary_row(2, v, i) for i, v in enumerate(vals)]
        p = write_csv(tmp_path / "n.csv", rows)
        nf = noise_floor(p)
        assert nf.n == 5
        assert nf.mean_ms == pytest.approx(100.0)
        assert nf.sd_ms == pytest.approx(1.5811, abs=1e-3)  # 標本標準偏差
        assert nf.cv_percent == pytest.approx(1.5811, abs=1e-3)

    def test_verdict_passes_when_signal_exceeds_two_sigma(self, tmp_path):
        rows = [boundary_row(2, 200.0 + 0.1 * i, i) for i in range(10)]
        p = write_csv(tmp_path / "q.csv", rows)
        nf = noise_floor(p)
        ok, msg = nf.verdict(expected_signal_ms=1.53)
        assert ok and "検出可能圏" in msg

    def test_verdict_fails_when_noise_dominates(self, tmp_path):
        rows = [boundary_row(2, 200.0 + 10.0 * (i % 2), i) for i in range(10)]
        p = write_csv(tmp_path / "loud.csv", rows)
        nf = noise_floor(p)
        ok, msg = nf.verdict(expected_signal_ms=1.53)
        assert not ok and "埋もれる" in msg


class TestSummarize:
    def test_groups_and_averages_repeats(self, tmp_path):
        rows = [
            {"sweep": "offload", "repeat": r, "n_gpu": n, "offload_ratio": n / 36,
             "crossings_analytic": 0 if n == 36 else 2,
             "decode_mean_ms": 400.0 - 9 * n + r, "tokens_per_sec": 2.0 + 0.1 * n,
             "ttft_ms": 1000.0, "peak_vram_mb": 1000 + 300 * n}
            for n in (0, 18, 36) for r in range(3)
        ]
        p = write_csv(tmp_path / "o.csv", rows)
        s = summarize_offload(p)
        assert [x["n_gpu"] for x in s] == [0, 18, 36]
        assert all(x["repeats"] == 3 for x in s)
        assert s[0]["decode_ms_mean"] == pytest.approx(401.0)  # (400+401+402)/3


class TestTTable:
    def test_small_df_uses_table(self):
        assert t95(2) == 4.303

    def test_large_df_falls_back_to_normal(self):
        assert t95(500) == 1.960

    def test_invalid_df(self):
        import math
        assert math.isnan(t95(0))


class TestBlockedRegression:
    """反復をブロック因子として扱う対応のある回帰（Step 1の実測で必要性が判明）。

    Step 1 では反復間に系統的なドリフトがあり、変動係数が5〜13%に達した。
    必要な分解能は約0.9%なので、ドリフトを除かないと傾きを検出できない。
    """

    @staticmethod
    def _rows_with_drift(true_slope=0.07, drift_per_repeat=20.0, repeats=5):
        """真の傾きは同じだが、反復ごとに大きなオフセット（ドリフト）が乗るデータ。

        ドリフトは境界数と無相関なので傾きの推定自体は不偏だが、
        単純な最小二乗では誤差項に入って標準誤差を膨らませ、有意にならない。
        """
        return [
            boundary_row(c, 250.0 + true_slope * c + drift_per_repeat * rep, rep)
            for rep in range(repeats)
            for c in (2, 6, 18, 36)
        ]

    def test_blocked_detects_what_pooled_cannot(self, tmp_path):
        from app.analyze import fit_boundary_cost_blocked

        p = write_csv(tmp_path / "drift.csv", self._rows_with_drift())

        pooled = fit_boundary_cost(p)
        blocked = fit_boundary_cost_blocked(p)

        # 傾きの点推定はどちらも真値に近い（ドリフトは境界数と無相関のため）
        assert pooled.slope == pytest.approx(0.07, abs=1e-6)
        assert blocked.slope == pytest.approx(0.07, abs=1e-6)

        # だが単純な最小二乗はドリフトに埋もれて有意にならない
        assert not pooled.significant
        # 対応分析ならドリフトが相殺され、検出できる
        assert blocked.significant
        # 信頼区間の幅も桁違いに狭くなる
        pw = pooled.ci95[1] - pooled.ci95[0]
        bw = blocked.ci95[1] - blocked.ci95[0]
        assert bw < pw / 100

    def test_blocked_reports_upper_bound_when_no_effect(self, tmp_path):
        """真の傾きが0なら、対応分析でも有意にならず上限値を示すこと。"""
        from app.analyze import fit_boundary_cost_blocked

        rows = self._rows_with_drift(true_slope=0.0)
        # 反復内にも小さなノイズを入れる（完全に0残差だと分散推定が退化するため）
        for i, r in enumerate(rows):
            r["decode_mean_ms"] += 0.5 * ((i % 3) - 1)
        p = write_csv(tmp_path / "noeffect.csv", rows)
        reg = fit_boundary_cost_blocked(p)
        assert reg.slope == pytest.approx(0.0, abs=0.01)
        assert not reg.significant
        assert "以下" in reg.report()

    def test_degrees_of_freedom_accounts_for_blocks(self, tmp_path):
        from app.analyze import fit_boundary_cost_blocked

        p = write_csv(tmp_path / "df.csv", self._rows_with_drift(repeats=5))
        reg = fit_boundary_cost_blocked(p)
        # n=20観測、5反復ぶんの平均と傾きを推定 -> df = 20 - 5 - 1
        assert reg.n == 20
        assert reg.df == 14

    def test_raises_when_crossings_constant(self, tmp_path):
        """各反復に複数点あるが境界数が全て同じ場合は回帰できない。"""
        from app.analyze import fit_boundary_cost_blocked

        rows = [
            boundary_row(2, 250.0 + r + 0.1 * i, r)
            for r in range(5)
            for i in range(4)
        ]
        p = write_csv(tmp_path / "const2.csv", rows)
        with pytest.raises(ValueError, match="回帰できない"):
            fit_boundary_cost_blocked(p)

    def test_raises_when_each_block_has_one_point(self, tmp_path):
        """1反復1点だとブロック内に情報が無く、対応分析は成立しない。"""
        from app.analyze import fit_boundary_cost_blocked

        rows = [boundary_row(c, 250.0 + 0.07 * c, r) for r, c in enumerate((2, 6, 18, 36))]
        p = write_csv(tmp_path / "single.csv", rows)
        with pytest.raises(ValueError, match="データが不足"):
            fit_boundary_cost_blocked(p)
