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


# ---------------------------------------------------------------------------
# Tube non-clustering (Katz-Tao Convex Wolff Axiom error) tests
# ---------------------------------------------------------------------------

def _star_segments(n=300):
    """Fibonacci-sphere directions, all segments through the origin."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = phi * i
    dirs = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return np.stack([-dirs * 0.5, dirs * 0.5], axis=1)


def _scattered_segments(n=300, seed=0):
    """Same directions, but each segment centre randomly translated."""
    rng = np.random.default_rng(seed)
    segs = _star_segments(n)
    centres = rng.uniform(-0.4, 0.4, size=(n, 3))
    return segs + centres[:, None, :]


def _tube_bundle(n=2000, length=0.3, region=0.3, seed=0):
    """Dense parallel tube bundle squeezed into a small central region."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-region / 2, region / 2, size=(n, 3))
    p0 = centres - np.array([0.0, 0.0, length / 2])
    p1 = centres + np.array([0.0, 0.0, length / 2])
    return np.stack([p0, p1], axis=1)


def _tube_uniform(n=2000, length=0.3, seed=1):
    """Same tubes, centres spread uniformly over the unit cube."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.5, 0.5, size=(n, 3))
    p0 = centres - np.array([0.0, 0.0, length / 2])
    p1 = centres + np.array([0.0, 0.0, length / 2])
    return np.stack([p0, p1], axis=1)


class TestTubeCKError:
    def test_contained_detects_clustering(self):
        """A dense parallel bundle violates the Wolff axioms (CKT > 1)."""
        from fracDimPy import tube_ck_error

        ck, diag = tube_ck_error(_tube_bundle(), delta=0.02, mode="contained", seed=42)
        assert ck > 1, f"bundle CKT={ck} should exceed 1 (clustered)"
        assert diag["n_tubes"] == 2000

    def test_contained_uniform_is_nonclustering(self):
        from fracDimPy import tube_ck_error

        ck, _ = tube_ck_error(_tube_uniform(), delta=0.02, mode="contained", seed=42)
        assert ck < 1, f"uniform CKT={ck} should be below 1 (non-clustering)"

    def test_contained_separates(self):
        """CKT of a clustered bundle should clearly exceed the uniform one."""
        from fracDimPy import tube_ck_error

        ck_b, _ = tube_ck_error(_tube_bundle(), delta=0.02, mode="contained", seed=42)
        ck_u, _ = tube_ck_error(_tube_uniform(), delta=0.02, mode="contained", seed=42)
        assert ck_b > 3 * ck_u, f"bundle {ck_b:.2f} should dominate uniform {ck_u:.2f}"

    def test_passing_detects_crossing_clustering(self):
        """All segments through a common point => high crossing density."""
        from fracDimPy import tube_ck_error

        ck_s, _ = tube_ck_error(_star_segments(), delta=0.01, mode="passing", seed=42)
        ck_t, _ = tube_ck_error(_scattered_segments(), delta=0.01, mode="passing", seed=42)
        assert ck_s > 5 * ck_t, f"star {ck_s:.1f} should dominate scattered {ck_t:.1f}"

    def test_invalid_mode(self):
        from fracDimPy import tube_ck_error

        with pytest.raises(ValueError):
            tube_ck_error(_star_segments(), delta=0.01, mode="nope")

    def test_invalid_shape(self):
        from fracDimPy import tube_ck_error

        with pytest.raises(ValueError):
            tube_ck_error(np.zeros((10, 2)), delta=0.01)  # not (K, 2, 3)

    def test_reproducible(self):
        from fracDimPy import tube_ck_error

        a, _ = tube_ck_error(_tube_uniform(), delta=0.02, seed=7)
        b, _ = tube_ck_error(_tube_uniform(), delta=0.02, seed=7)
        assert a == b
