# Side-by-side vegetation-on vs vegetation-off comparison (Suppl. Fig. 2 style).

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(outdir: str, base: str, name: str) -> np.ndarray:
    return np.load(f"{outdir}/{base}_{name}.npy")


def compare(outdir_on: str, outdir_off: str, run: int = 9, subrun: str = "SUBRUNa1_SUBRUNb1") -> str:
    base = f"ComplexChannelPatterns_Fig1f_RUN{run}_{subrun}"
    s_on = _load(outdir_on, base, "s")
    s_off = _load(outdir_off, base, "s")
    d_on = _load(outdir_on, base, "d")
    b_on = _load(outdir_on, base, "b")
    b_off = _load(outdir_off, base, "b")

    # Bed elevation change relative to the initial flat bed (erosion < 0)
    ero_on = s_on + b_on - b_on.mean() - (s_on + b_on)
    z_on = s_on + b_on
    z_off = s_off + b_off
    relief_on = z_on.max() - z_on.min()
    relief_off = z_off.max() - z_off.min()
    veg_fraction = float((d_on > 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (z, title) in zip(
        axes,
        ((z_off, f"without vegetation (relief {relief_off:.3f} m)"),
         (z_on, f"with vegetation (relief {relief_on:.3f} m, veg cover {veg_fraction:.1%})")),
    ):
        im = ax.imshow(z, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

    out = f"{outdir_on}/comparison_veg_on_off.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(out)
    print(f"veg-on relief: {relief_on:.4f} m | veg-off relief: {relief_off:.4f} m | ratio: {relief_on / max(relief_off, 1e-12):.2f}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare veg-on vs veg-off runs (Suppl. Fig. 2 style)")
    p.add_argument("--on", default="RUN9", help="output dir of the vegetation-on run")
    p.add_argument("--off", default="RUN9_vegoff", help="output dir of the vegetation-off run")
    p.add_argument("--run", type=int, default=9)
    args = p.parse_args()
    compare(args.on, args.off, args.run)