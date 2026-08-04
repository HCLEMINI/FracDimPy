#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Divider (Richardson) and Minkowski-MC dimension methods.

Both are intended for sparse-line / network sets where box-counting is
ill-conditioned. Verified on known-dimension references:
- 2D Koch curve, true D = log4/log3 ~ 1.2619
- 3D straight line, true D = 1
"""

import numpy as np
import pytest
from fracDimPy import (
    divider_dimension,
    minkowski_dimension_mc,
    generate_koch_curve,
)


class TestDividerDimension:
    def test_koch_curve(self):
        pts, _ = generate_koch_curve(level=6, size=512)
        D, res = divider_dimension(pts)
        assert 1.20 < D < 1.32, f"Koch divider D={D} should be ~1.262"
        assert res["R2"] > 0.95

    def test_3d_line(self):
        """A straight line has D=1; box-counting fails on this, divider does not."""
        t = np.linspace(0, 1, 2000)[:, None]
        line = np.hstack([t, t, t])
        D, res = divider_dimension(line)
        assert 0.95 < D < 1.05, f"3D line divider D={D} should be ~1.0"
        assert res["R2"] > 0.98

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            divider_dimension(np.zeros((2, 2)))

    def test_result_keys(self):
        pts, _ = generate_koch_curve(level=5, size=256)
        _, res = divider_dimension(pts)
        for k in ("dimension", "delta_values", "N_values", "R2", "method"):
            assert k in res

    def test_silent_by_default(self, capsys):
        pts, _ = generate_koch_curve(level=5, size=256)
        divider_dimension(pts)
        assert capsys.readouterr().out == ""


class TestMinkowskiMCDimension:
    def test_koch_curve(self):
        pts, _ = generate_koch_curve(level=6, size=512)
        D, res = minkowski_dimension_mc(pts, seed=42)
        assert 1.15 < D < 1.35, f"Koch Minkowski D={D} should be ~1.262"
        assert res["R2"] > 0.9

    def test_3d_line(self):
        t = np.linspace(0, 1, 2000)[:, None]
        line = np.hstack([t, t, t])
        D, res = minkowski_dimension_mc(line, seed=42)
        assert 0.8 < D < 1.2, f"3D line Minkowski D={D} should be ~1.0"
        assert res["R2"] > 0.9

    def test_reproducible(self):
        pts, _ = generate_koch_curve(level=6, size=512)
        a, _ = minkowski_dimension_mc(pts, seed=7)
        b, _ = minkowski_dimension_mc(pts, seed=7)
        assert a == b

    def test_auto_transpose(self):
        pts, _ = generate_koch_curve(level=6, size=512)
        D, res = minkowski_dimension_mc(pts.T, seed=42)  # (2, M) -> (M, 2)
        assert 1.15 < D < 1.35
        assert res["embedding_dim"] == 2

    def test_zero_extent_raises(self):
        with pytest.raises(ValueError):
            minkowski_dimension_mc(np.zeros((100, 2)))
