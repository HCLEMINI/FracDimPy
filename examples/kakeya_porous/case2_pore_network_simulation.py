#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case 2 — Simulating pore-throat networks (algorithm exploration)
================================================================

A pore-throat network models a porous medium as pores (bodies) connected by
throats (narrow channels). The throat centre lines form a set of tubes, which
is exactly the input our Kakeya machinery consumes.

Algorithm selection (explored here)
-----------------------------------
1. **Voronoi network** (implemented, the digital-rock standard): scatter pore
   seeds at random; the Voronoi partition of the seed points is the dual of a
   sphere-packing / grain model — Voronoi edges inside the box are the throat
   centre lines. Naturally isotropic (directions uniformly spread) and
   non-clustered, i.e. a "healthy" network baseline.
2. Alternatives noted for later exploration:
   - random sphere packing + throat connect (grain-scale realism, costlier);
   - lattice perturbation (regular grid + jitter, cheap, less realistic);
   - power-law / fractal throats (long-tailed throat lengths, mimics
     fractal pore media); a direction-biased variant of Voronoi (anisotropic
     permeability) can be obtained by anisotropic scaling of the seeds.

Run:    python case2_pore_network_simulation.py
Output: output/case2_pore_network.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

from fracDimPy import box_counting

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def generate_pore_network_3d(n_seeds=150, box_size=1.0, seed=None, center_box=False):
    """3D Voronoi pore-throat network.

    Parameters
    ----------
    n_seeds : int
        Number of pore seeds (randomly placed grains).
    box_size : float
        Side length of the cubic domain.
    seed : int, optional
        Reproducibility.
    center_box : bool
        Shift the network so the box is centred at the origin.

    Returns
    -------
    np.ndarray
        Throat centre lines, shape (K, 2, 3).
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, box_size, size=(n_seeds, 3))
    vor = Voronoi(pts)

    segs = []
    for r in vor.ridge_vertices:
        if -1 in r:
            continue  # unbounded ridge
        p0, p1 = vor.vertices[r[0]], vor.vertices[r[1]]
        inside = (
            (p0 >= 0.0).all() and (p1 >= 0.0).all()
            and (p0 <= box_size).all() and (p1 <= box_size).all()
        )
        if inside:
            segs.append([p0, p1])
    segs = np.asarray(segs, dtype=float)
    if center_box:
        segs -= box_size / 2.0
    return segs


def sample_throat_cloud(segs, points_per_throat=20):
    """Sample points along each throat centre line -> point cloud for box-counting."""
    s = np.linspace(0.0, 1.0, points_per_throat)[None, :, None]
    pts = (segs[:, 0][:, None, :] + (segs[:, 1] - segs[:, 0])[:, None, :] * s)
    return pts.reshape(-1, 3)


def main():
    rng = np.random.default_rng(0)
    segs = generate_pore_network_3d(n_seeds=150, box_size=1.0, seed=0, center_box=True)

    # ---- structure statistics ----
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    dirs = (segs[:, 1] - segs[:, 0]) / L[:, None]
    print(f"throats        : {len(segs)}")
    print(f"mean length    : {L.mean():.4f} +/- {L.std():.4f}")
    print(f"length range   : [{L.min():.4f}, {L.max():.4f}]")
    # isotropy check: mean |direction component| close to 0.5 for uniform spread
    print(f"mean |dir comp|: {np.abs(dirs).mean(axis=0).round(3)}  (0.5 = isotropic)")

    # ---- box-counting dimension of the sampled throat cloud ----
    cloud = sample_throat_cloud(segs)
    D, res = box_counting(cloud, data_type="points")
    print(f"box-counting D : {D:.4f}  (R2={res['R2']:.4f})")

    # ---- figures ----
    fig = plt.figure(figsize=(13, 4.4))
    # (a) 3D network (subset)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for s in segs[:200]:
        ax.plot(s[:, 0], s[:, 1], s[:, 2], lw=0.7, color="#156082")
    ax.set_title(f"Voronoi pore-throat network ({len(segs)} throats)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # (b) throat length histogram
    ax = fig.add_subplot(1, 3, 2)
    ax.hist(L, bins=30, color="#156082", alpha=0.8, edgecolor="white")
    ax.set_xlabel("throat length")
    ax.set_ylabel("count")
    ax.set_title("Throat length distribution")
    ax.grid(alpha=0.3)

    # (c) log-log box-counting fit
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(res["log_inv_epsilon"], res["log_N"], "o", ms=5, color="#156082", alpha=0.75)
    x = np.linspace(min(res["log_inv_epsilon"]), max(res["log_inv_epsilon"]), 50)
    ax.plot(x, np.polyval(res["coefficients"], x), "-", color="#C00000", lw=1.8)
    ax.set_xlabel("log(1/$\\varepsilon$)")
    ax.set_ylabel("log N($\\varepsilon$)")
    ax.set_title(f"Network dimension: D={D:.3f}, R$^2$={res['R2']:.4f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT, "case2_pore_network.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
