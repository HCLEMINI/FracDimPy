#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case 3 — Engineering feasibility: can D + CKT separate good from bad networks?
==============================================================================

Four synthetic throat networks are compared with the two structural indicators
now in the library:

- D          : box-counting dimension of the sampled throat cloud
               (directional coverage of the network)
- CKT        : Katz-Tao Wolff-axiom error, tube-body clustering
               (local enrichment / bottleneck risk)

Networks:
  1. voronoi        — isotropic, uniformly spread (healthy baseline)
  2. kakeya         — one throat per direction (theoretical ideal)
  3. unidirectional — all throats parallel (directional coverage missing)
  4. clustered      — parallel throats squeezed into a small region (local
                      enrichment, the "bottleneck" pathology)

Run:    python case3_ck_dimension_feasibility.py
Output: output/case3_feasibility.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from case2_pore_network_simulation import generate_pore_network_3d, sample_throat_cloud
from fracDimPy import box_counting, tube_ck_error, kakeya_segments

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

DELTA = 0.01  # nominal throat radius, kept identical across networks for fairness


def unidirectional_bundle(n=2000, length=0.3, seed=0):
    """Parallel throats, positions uniform — direction coverage missing."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.5, 0.5, size=(n, 3))
    p0 = centres - np.array([0.0, 0.0, length / 2])
    p1 = centres + np.array([0.0, 0.0, length / 2])
    return np.stack([p0, p1], axis=1)


def clustered_bundle(n=2000, length=0.3, region=0.2, seed=0):
    """Parallel throats squeezed into a small region — local enrichment."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-region / 2, region / 2, size=(n, 3))
    p0 = centres - np.array([0.0, 0.0, length / 2])
    p1 = centres + np.array([0.0, 0.0, length / 2])
    return np.stack([p0, p1], axis=1)


def evaluate(name, segs):
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    dirs = (segs[:, 1] - segs[:, 0]) / L[:, None]
    anisotropy = float(np.abs(dirs).mean(axis=0).max())  # 0.5 isotropic, 1.0 single-direction

    cloud = sample_throat_cloud(segs, points_per_throat=20)
    D, res = box_counting(cloud, data_type="points")
    ck_c, _ = tube_ck_error(segs, DELTA, mode="contained", seed=42)
    ck_p, _ = tube_ck_error(segs, DELTA, mode="passing", seed=42)

    print(
        f"{name:>14s} | throats={len(segs):5d} | D={D:5.3f} (R2={res['R2']:.3f}) | "
        f"CKT_cont={ck_c:6.2f} | CKT_pass={ck_p:7.1f} | anisotropy={anisotropy:.2f}"
    )
    return dict(name=name, segs=segs, D=D, ck_c=ck_c, ck_p=ck_p, aniso=anisotropy)


def main():
    print(f"{'network':>14s} | {'n':>5s} | {'D':>9s} | {'CKT_contained':>13s} | "
          f"{'CKT_passing':>11s} | anisotropy")
    print("-" * 88)

    results = []
    results.append(evaluate("voronoi", generate_pore_network_3d(n_seeds=150, seed=0, center_box=True)))
    results.append(evaluate("kakeya", kakeya_segments(3, 2000, 1.0)))
    results.append(evaluate("unidirectional", unidirectional_bundle(seed=0)))
    results.append(evaluate("clustered", clustered_bundle(seed=0)))

    print()
    print("interpretation (three indicators jointly):")
    print("  - anisotropy   : 0.5 isotropic, 1.0 single-direction (directional coverage)")
    print("  - CKT_contained: tube-body clustering / local enrichment (bottleneck risk)")
    print("  - D            : spatial filling of the throat cloud")
    print()
    for r in results:
        issues = []
        if r["aniso"] > 0.7:
            issues.append("directional coverage MISSING (anisotropic)")
        if r["ck_c"] > 1.0:
            issues.append("local enrichment / bottleneck risk (CKT>1)")
        if r["D"] < 1.8:
            issues.append("sparse filling (D low)")
        status = ", ".join(issues) if issues else "healthy (isotropic + non-clustered)"
        print(f"  {r['name']:>14s}: {status}")

    # ---- figure: D vs CKT (contained), labelled by network ----
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {"voronoi": "#156082", "kakeya": "#2E8B57", "unidirectional": "#C00000",
              "clustered": "#8B4513"}
    for r in results:
        ax.scatter(r["ck_c"], r["D"], s=110, color=colors[r["name"]], zorder=3, edgecolor="white")
        ax.annotate(r["name"], (r["ck_c"], r["D"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.axvspan(0, 2.0, color="#2E8B57", alpha=0.08)
    ax.text(0.05, 0.04, "healthy band (CKT < 2)", transform=ax.transAxes, fontsize=9,
            color="#2E8B57")
    ax.set_xscale("log")
    ax.set_xlabel("CKT (tube-body clustering, log scale)")
    ax.set_ylabel("box-counting dimension D")
    ax.set_title("Structural health map: D vs CKT")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT, "case3_feasibility.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
