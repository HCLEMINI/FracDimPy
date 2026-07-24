#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Box-counting on Point Clouds (data_type="points")
=====================================================

Verifies the memory-light point-cloud box counting path:
- Strict equivalence with dense fixed-grid counting at equal epsilon
- Reasonable dimension estimates on random 1D/3D clouds
- Auto-transpose, silence, and error handling
"""

import numpy as np
import pytest
from fracDimPy import box_counting
from fracDimPy.utils.box_counting_core import count_boxes_points, count_nonempty


class TestBoxCountingPoints:
    @pytest.fixture
    def sparse_3d(self):
        rng = np.random.default_rng(42)
        arr = (rng.random((64, 64, 64)) < 0.05).astype(np.int8)
        coords = np.argwhere(arr > 0).astype(float)
        return arr, coords

    def test_equivalence_with_dense(self, sparse_3d):
        """At equal epsilon, points and dense fixed-grid count identical boxes."""
        arr, coords = sparse_3d
        for eps in [2, 4, 8, 16, 32]:
            assert count_nonempty(arr, eps) == count_boxes_points(coords, float(eps))

    def test_3d_random_dimension(self):
        rng = np.random.default_rng(42)
        arr = (rng.random((96, 96, 96)) < 0.04).astype(np.int8)
        coords = np.argwhere(arr > 0).astype(float)
        D, res = box_counting(coords, data_type="points")
        assert 2.4 < D < 3.0, f"3D random cloud D={D} should be near 3"
        assert res["R2"] > 0.95

    def test_1d_dimension(self):
        x = np.random.default_rng(1).standard_normal(3000)
        D, res = box_counting(x, data_type="points")
        assert 0.8 < D < 1.2, f"1D cloud D={D} should be near 1"
        assert res["R2"] > 0.95

    def test_auto_transpose(self):
        # A (d, M) array should be interpreted as M points in d dims
        pts = np.random.default_rng(2).uniform(-1, 1, (3, 20000))
        D, res = box_counting(pts, data_type="points")
        assert 2.4 < D < 3.0
        assert res["embedding_dim"] == 3

    def test_silent_by_default(self, capsys):
        pts = np.random.default_rng(3).standard_normal((1000, 2))
        box_counting(pts, data_type="points")
        assert capsys.readouterr().out == ""

    def test_verbose_prints(self, capsys):
        pts = np.random.default_rng(4).standard_normal((500, 2))
        box_counting(pts, data_type="points", verbose=True)
        assert capsys.readouterr().out != ""

    def test_invalid_data_type(self):
        with pytest.raises(ValueError):
            box_counting(np.zeros(10), data_type="not_a_type")

    def test_zero_extent_raises(self):
        # All identical points -> zero extent
        pts = np.zeros((100, 2))
        with pytest.raises(ValueError):
            box_counting(pts, data_type="points")
