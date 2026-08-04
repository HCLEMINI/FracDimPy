#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case 1 — Numerical illustration of the 3D Kakeya conjecture (Wang & Zahl 2025)
==============================================================================

Generates discrete 3D Kakeya sets with increasing direction counts and shows
that the box-counting (Minkowski) dimension converges toward 3 — a numerical
illustration of the theorem, *not* a proof: a finite, discretised set always
sits below d, and the gap closes only as direction count and sampling density
grow.

Run:    python case1_kakeya_dimension_3d.py
Output: output/case1_kakeya_dim_3d.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracDimPy import generate_kakeya_set, box_counting

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def kakeya_segments_2d(n):
    """2D star segments (all through the origin), one per direction."""
    theta = np.linspace(0.0, np.pi, n, endpoint=False)
    d = np.column_stack([np.cos(theta), np.sin(theta)])
    return np.stack([-d * 0.5, d * 0.5], axis=1)


def main():
    ndirs = [250, 500, 1000, 2000, 4000, 8000, 16000]
    Ds, R2s = [], []
    print(f"{'num_directions':>15} {'D':>8} {'R2':>8}")
    for nd in ndirs:
        pts = generate_kakeya_set(
            dimension=3, num_directions=nd, length=1.0, points_per_segment=120, seed=42
        )
        D, res = box_counting(pts, data_type="points")
        Ds.append(D)
        R2s.append(res["R2"])
        print(f"{nd:>15} {D:8.4f} {res['R2']:8.4f}")

    # detailed log-log fit at 4000 directions
    pts4 = generate_kakeya_set(
        dimension=3, num_directions=4000, length=1.0, points_per_segment=120, seed=42
    )
    D4, r4 = box_counting(pts4, data_type="points")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) dimension convergence toward 3
    ax = axes[0]
    ax.semilogx(ndirs, Ds, "o-", color="#C00000", lw=1.8, ms=5)
    ax.axhline(3.0, ls="--", color="k", alpha=0.6, lw=1.2)
    ax.text(
        0.98, 0.10, "target D = 3 (Wang & Zahl 2025)",
        transform=ax.transAxes, ha="right", fontsize=9, color="#333333",
    )
    ax.set_xlabel("num_directions (log scale)")
    ax.set_ylabel("box-counting dimension D")
    ax.set_title("3D Kakeya: D converges toward 3")
    ax.set_ylim(2.0, 3.1)
    ax.grid(alpha=0.3)

    # (b) log-log box-counting fit
    ax = axes[1]
    ax.plot(r4["log_inv_epsilon"], r4["log_N"], "o", ms=5, color="#156082", alpha=0.75)
    x = np.linspace(min(r4["log_inv_epsilon"]), max(r4["log_inv_epsilon"]), 50)
    ax.plot(x, np.polyval(r4["coefficients"], x), "-", color="#C00000", lw=1.8)
    ax.set_xlabel("log(1/$\\varepsilon$)")
    ax.set_ylabel("log N($\\varepsilon$)")
    ax.set_title(f"Box-counting fit, 4000 dirs: D={D4:.3f}, R$^2$={r4['R2']:.4f}")
    ax.grid(alpha=0.3)

    # (c) 2D Kakeya illustration (60 of 200 segments)
    ax = axes[2]
    segs2 = kakeya_segments_2d(200)
    for s in segs2[:60]:
        ax.plot(s[:, 0], s[:, 1], lw=0.5, color="#156082")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("2D Kakeya set (60 of 200 segments)")

    fig.tight_layout()
    out = os.path.join(OUT, "case1_kakeya_dim_3d.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"\nfigure saved: {out}")
    print(
        f"\nconclusion: D rises {Ds[0]:.3f} -> {Ds[-1]:.3f} as direction count grows; "
        f"finite-direction discreteness keeps D < 3. The trend confirms the theorem's "
        f"limit D -> 3 in a numerical sense (Minkowski side)."
    )


if __name__ == "__main__":
    main()
