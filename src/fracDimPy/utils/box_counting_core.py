#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core box counting primitives for fractal analysis.

Provides dimension-agnostic box counting functions using np.add.reduceat,
replacing duplicated implementations across monofractal and multifractal modules.
"""

import numpy as np


def count_boxes_fixed(MT, epsilon):
    """Fixed-grid box counting for any dimension.

    Uses np.add.reduceat along each axis to aggregate boxes.

    Parameters
    ----------
    MT : numpy.ndarray
        Binary or integer matrix (1D, 2D, or 3D).
    epsilon : int
        Box size.

    Returns
    -------
    numpy.ndarray
        Aggregated box values (same ndim as MT, reduced size).
    """
    result = MT  # np.add.reduceat returns a new array; no copy is needed.
    for axis in range(MT.ndim):
        indices = np.arange(0, MT.shape[axis], epsilon)
        if len(indices) == 0:
            continue
        result = np.add.reduceat(result, indices, axis=axis)
    return result


def count_nonempty(MT, epsilon, max_fill=None):
    """Count non-empty boxes using fixed-grid strategy.

    Parameters
    ----------
    MT : numpy.ndarray
        Binary or integer matrix.
    epsilon : int
        Box size.
    max_fill : float, optional
        Maximum expected fill per box. If None, uses epsilon**ndim.

    Returns
    -------
    int
        Number of non-empty boxes.
    """
    aggregated = count_boxes_fixed(MT, epsilon)
    if max_fill is None:
        max_fill = epsilon ** MT.ndim
    return int(np.sum((aggregated > 0) & (aggregated <= max_fill)))


def count_boxes_points(coords, epsilon):
    """Count non-empty boxes covering a point cloud at scale ``epsilon``.

    Memory-light alternative to dense box counting: instead of allocating an
    ``N**d`` grid, each point is hashed to its enclosing box and the number of
    distinct boxes is counted. Mathematically equivalent to dense fixed-grid
    box counting on the same point set, but uses ``O(M)`` memory for ``M``
    points instead of ``O(N**d)``.

    Parameters
    ----------
    coords : numpy.ndarray
        Point coordinates, shape ``(M,)`` or ``(M, d)``.
    epsilon : float
        Box size, in the same units as ``coords``.

    Returns
    -------
    int
        Number of distinct boxes that contain at least one point.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    boxes = np.floor(coords / epsilon).astype(np.int64)
    boxes = np.ascontiguousarray(boxes)
    # Fast row-wise uniqueness via a void-typed view of the raw bytes; avoids
    # both Python-level encoding and the overflow risk of manual 1D hashing.
    view = boxes.view(np.dtype((np.void, boxes.dtype.itemsize * boxes.shape[1])))
    return int(np.unique(view).shape[0])


def count_boxes_sliding(MT, epsilon, step=None, max_fill=None):
    """Sliding window box counting for any dimension.

    Parameters
    ----------
    MT : numpy.ndarray
        Binary or integer matrix.
    epsilon : int
        Box size.
    step : float, optional
        Sliding step as fraction of epsilon (default 0.5).
    max_fill : float, optional
        Maximum expected fill per box.

    Returns
    -------
    float
        Adjusted count of non-empty boxes.
    """
    if step is None:
        step = 0.5
    step_size = max(1, int(epsilon * step))
    if max_fill is None:
        max_fill = epsilon ** MT.ndim

    total_count = 0
    total_positions = 0

    # Generate offsets for each axis
    offsets_per_axis = [range(0, epsilon, step_size) for _ in range(MT.ndim)]

    # Iterate over all offset combinations
    import itertools
    for offsets in itertools.product(*offsets_per_axis):
        # Slice the array with the offset
        slices = tuple(slice(o, None) for o in offsets)
        sub = MT[slices]

        # Ensure subarray can fit at least one box
        if any(s < epsilon for s in sub.shape):
            continue

        aggregated = count_boxes_fixed(sub, epsilon)
        count = int(np.sum((aggregated > 0) & (aggregated <= max_fill)))
        total_count += count
        total_positions += 1

    if total_positions > 0:
        overlap_factor = (epsilon / step_size) ** MT.ndim
        return total_count / (total_positions * overlap_factor) * (step_size / epsilon) ** MT.ndim
    return 0


def count_boxes_random(MT, epsilon, n_random=5, max_fill=None):
    """Random position sampling box counting.

    Parameters
    ----------
    MT : numpy.ndarray
        Binary or integer matrix.
    epsilon : int
        Box size.
    n_random : int, optional
        Number of random positions to sample (default 5).
    max_fill : float, optional
        Maximum expected fill per box.

    Returns
    -------
    float
        Estimated count of non-empty boxes.
    """
    if max_fill is None:
        max_fill = epsilon ** MT.ndim

    counts = []
    for _ in range(n_random):
        # Random offset for each axis
        offsets = [np.random.randint(0, max(1, epsilon)) for _ in range(MT.ndim)]
        slices = tuple(slice(o, None) for o in offsets)
        sub = MT[slices]

        if any(s < epsilon for s in sub.shape):
            continue

        aggregated = count_boxes_fixed(sub, epsilon)
        count = int(np.sum((aggregated > 0) & (aggregated <= max_fill)))
        counts.append(count)

    if not counts:
        return 0

    # Scale up from sampled region to full array
    avg_count = np.mean(counts)
    scale_factors = [(s - epsilon + 1) / max(1, s - epsilon)
                     for s in MT.shape if s > epsilon]
    scale = np.prod(scale_factors) if scale_factors else 1
    return avg_count * scale


def count_boxes_advanced(MT, epsilon, strategy="fixed", n_random=5, sliding_step=0.5, max_fill=None):
    """Dispatch box counting to the appropriate strategy.

    Parameters
    ----------
    MT : numpy.ndarray
        Binary or integer matrix.
    epsilon : int
        Box size.
    strategy : str, optional
        Counting strategy: 'fixed', 'sliding', or 'random'.
    n_random : int, optional
        Number of random samples (for 'random' strategy).
    sliding_step : float, optional
        Step fraction for 'sliding' strategy.
    max_fill : float, optional
        Maximum expected fill per box.

    Returns
    -------
    int or float
        Number (or estimate) of non-empty boxes.
    """
    if strategy == "fixed":
        return count_nonempty(MT, epsilon, max_fill)
    elif strategy == "sliding":
        return count_boxes_sliding(MT, epsilon, sliding_step, max_fill)
    elif strategy == "random":
        return count_boxes_random(MT, epsilon, n_random, max_fill)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed', 'sliding', or 'random'.")
