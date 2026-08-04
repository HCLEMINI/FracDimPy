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
        assert 1.15 < D < 1.40, f"Koch Minkowski D={D} should be ~1.262"
        assert res["R2"] > 0.9

    def test_filled_2d_sets(self):
        """Minkowski-MC recovers known D on classic filled sets (after
        automatic scale-region detection)."""
        from fracDimPy import (
            generate_sierpinski, generate_sierpinski_carpet, generate_vicsek_fractal,
        )
        cases = [
            (generate_sierpinski(level=6, size=512), 1.585, 0.12),
            (generate_sierpinski_carpet(level=5, size=243), 1.893, 0.12),
            (generate_vicsek_fractal(level=5, size=243), 1.465, 0.18),
        ]
        for img, Dtrue, tol in cases:
            pts = np.argwhere(img > 0).astype(float)
            D, _ = minkowski_dimension_mc(pts, seed=42)
            assert abs(D - Dtrue) < tol, f"Minkowski D={D} vs true {Dtrue}"

    def test_menger_3d(self):
        from fracDimPy import generate_menger_sponge
        sp = generate_menger_sponge(level=3, size=27)
        pts = np.argwhere(sp > 0).astype(float)
        D, _ = minkowski_dimension_mc(pts, seed=42)
        assert abs(D - 2.727) < 0.2, f"Menger D={D} vs 2.727"

    def test_3d_line(self):
        t = np.linspace(0, 1, 2000)[:, None]
        line = np.hstack([t, t, t])
        D, res = minkowski_dimension_mc(line, seed=42)
        assert 0.8 < D < 1.2, f"3D line Minkowski D={D} should be ~1.0"
        assert res["R2"] > 0.9

    def test_self_affine_caveat(self):
        """Divider dimension on a self-affine curve (fBm) differs from its box
        dimension — a known mathematical fact, not a bug. This test documents
        the boundary: divider is for self-similar curves (Koch), not fBm."""
        from fracDimPy import generate_fbm_curve
        curve, _ = generate_fbm_curve(dimension=1.5, length=4096, seed=42)
        pts = np.column_stack([np.arange(len(curve)), curve])
        rng = np.ptp(pts, axis=0)
        rng[rng == 0] = 1
        pts = (pts - pts.min(axis=0)) / rng
        D, _ = divider_dimension(pts)
        # divider overestimates self-affine dim; just assert it runs and is in range
        assert 1.4 < D < 2.0

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
