#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Kakeya Set Generator
========================

A Kakeya set contains a unit line segment in every direction. Its Hausdorff and
Minkowski dimensions equal the embedding dimension d (Kakeya conjecture; proved
for d=2 by Davies 1971 and d=3 by Wang and Zahl 2025). These tests check the
generator shape/contract and that the discrete box-counting dimension of the
generated cloud is high (toward d) -- a numerical illustration, not a proof.
"""

import numpy as np
import pytest
from fracDimPy import generate_kakeya_set, box_counting


class TestKakeyaSet:
    def test_shape_2d(self):
        pts = generate_kakeya_set(dimension=2, num_directions=100, points_per_segment=50)
        assert pts.shape == (5000, 2)

    def test_shape_3d(self):
        pts = generate_kakeya_set(dimension=3, num_directions=100, points_per_segment=50)
        assert pts.shape == (5000, 3)

    def test_length_scales_extent(self):
        pts_small = generate_kakeya_set(dimension=2, num_directions=20, length=1.0, points_per_segment=50)
        pts_large = generate_kakeya_set(dimension=2, num_directions=20, length=4.0, points_per_segment=50)
        assert np.ptp(pts_large, axis=0).max() > np.ptp(pts_small, axis=0).max()

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            generate_kakeya_set(dimension=4)

    def test_invalid_dimension_negative(self):
        with pytest.raises(ValueError):
            generate_kakeya_set(dimension=1)

    def test_reproducible(self):
        """The default star construction is deterministic."""
        a = generate_kakeya_set(dimension=3, num_directions=50, points_per_segment=10, seed=42)
        b = generate_kakeya_set(dimension=3, num_directions=50, points_per_segment=10, seed=7)
        # seed does not change the deterministic star geometry
        assert np.array_equal(a, b)

    def test_2d_dimension_is_high(self):
        """Discrete 2D Kakeya box-counting dim sits clearly above 1 (toward 2)."""
        pts = generate_kakeya_set(
            dimension=2, num_directions=1500, length=1.0, points_per_segment=400, seed=42
        )
        D, res = box_counting(pts, data_type="points")
        assert D > 1.5, f"2D Kakeya D={D} should be well above 1"
        assert res["R2"] > 0.95

    def test_3d_dimension_is_high(self):
        """Discrete 3D Kakeya box-counting dim sits clearly above 2 (toward 3)."""
        pts = generate_kakeya_set(
            dimension=3, num_directions=2000, length=1.0, points_per_segment=120, seed=42
        )
        D, res = box_counting(pts, data_type="points")
        assert D > 2.2, f"3D Kakeya D={D} should be well above 2"
        assert res["R2"] > 0.95

    def test_3d_dimension_rises_with_directions(self):
        """More directions -> equal or higher estimated dimension."""
        pts_few = generate_kakeya_set(dimension=3, num_directions=500, points_per_segment=120, seed=42)
        pts_many = generate_kakeya_set(dimension=3, num_directions=4000, points_per_segment=120, seed=42)
        D_few, _ = box_counting(pts_few, data_type="points")
        D_many, _ = box_counting(pts_many, data_type="points")
        assert D_many >= D_few - 0.1, f"D should not drop with more directions: {D_few} -> {D_many}"
