"""placement.py の単体テスト。

境界数の解析的算出（基本設計4.4節）が設計書の表と一致することを固定する。
実モデルを必要としないため、CI的に高速に回せる。
"""

import pytest

from app.placement import (
    CPU,
    CUDA,
    boundary_sweep,
    build_blocked,
    build_contiguous,
    build_explicit,
    build_placement,
    offload_sweep,
)

N = 36  # Qwen2.5-3B の層数


class TestContiguous:
    def test_all_cpu(self):
        p = build_contiguous(N, 0)
        assert p.n_gpu == 0
        assert p.offload_ratio == 0.0
        assert set(p.layers) == {CPU}

    def test_all_gpu(self):
        p = build_contiguous(N, N)
        assert p.n_gpu == N
        assert p.offload_ratio == 1.0
        # 全層GPU、埋め込みも出力もGPUなので境界は生じない
        assert p.crossings == 0

    def test_all_cpu_crossings(self):
        # embed(GPU)→layer0(CPU) で1回、layer35(CPU)→norm(GPU) で1回
        assert build_contiguous(N, 0).crossings == 2

    def test_half(self):
        p = build_contiguous(N, 18)
        assert p.pattern == "G" * 18 + "C" * 18
        assert p.offload_ratio == 0.5

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            build_contiguous(N, N + 1)
        with pytest.raises(ValueError):
            build_contiguous(N, -1)


class TestBlocked:
    @pytest.mark.parametrize(
        ("block_size", "expected_crossings"),
        [(18, 2), (6, 6), (2, 18), (1, 36)],
    )
    def test_design_doc_table(self, block_size, expected_crossings):
        """基本設計4.4節の表（L1〜L4）と一致すること。"""
        p = build_blocked(N, block_size)
        assert p.crossings == expected_crossings
        assert p.offload_ratio == 0.5, "境界数スイープはオフロード率50%固定"

    def test_alternating_pattern(self):
        p = build_blocked(N, 1)
        assert p.pattern.startswith("GCGC")
        assert p.pattern.endswith("GC")

    def test_block_of_two(self):
        p = build_blocked(N, 2)
        assert p.pattern.startswith("GGCCGG")

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            build_blocked(N, 0)


class TestExplicit:
    def test_roundtrip(self):
        layers = [CUDA, CPU, CUDA]
        p = build_explicit(layers)
        assert p.layers == tuple(layers)
        assert p.n_layers == 3

    def test_rejects_unknown_device(self):
        with pytest.raises(ValueError):
            build_explicit([CUDA, "npu"])


class TestSweeps:
    def test_offload_sweep_endpoints(self):
        ps = offload_sweep(N, step=4)
        assert ps[0].n_gpu == 0
        assert ps[-1].n_gpu == N, "全層GPUの点は必ず含める"
        assert len(ps) == 10

    def test_offload_sweep_includes_last_when_not_divisible(self):
        ps = offload_sweep(10, step=4)
        assert [p.n_gpu for p in ps] == [0, 4, 8, 10]

    def test_boundary_sweep_ratio_is_constant(self):
        ps = boundary_sweep(N)
        assert {p.offload_ratio for p in ps} == {0.5}
        assert [p.crossings for p in ps] == [2, 6, 18, 36]

    def test_boundary_sweep_rejects_uneven_layers(self):
        # 層数が block_size*2 で割り切れないとオフロード率がずれるため弾く
        with pytest.raises(ValueError):
            boundary_sweep(10, block_sizes=(1, 3))


class TestDispatcher:
    def test_build_placement_contiguous(self):
        assert build_placement("contiguous", N, n_gpu=5).n_gpu == 5

    def test_build_placement_blocked(self):
        assert build_placement("blocked", N, block_size=2).crossings == 18

    def test_unknown_policy(self):
        with pytest.raises(ValueError):
            build_placement("magic", N)
