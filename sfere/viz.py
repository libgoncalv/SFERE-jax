# Visualization: 8-panel figure, MP4 animation, mean/min/max time series.
# Port of the plotting sections of the original notebook.

from __future__ import annotations

import os

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .config import RunConfig

_FS = 12
_TITLES = (
    ("s", "Sediment layer thickness [m]"),
    ("speed", "Depth-averaged flow speed [m/s]"),
    ("d", "Vegetation stem density [m^-2]"),
    ("b", "Subsediment bed topography [m]"),
    ("h", "Water layer thickness [m]"),
    ("c", "Chézy coefficient [m^(1/2)/s]"),
    ("t", "Bed shear stress [N/m2]"),
    ("eta", "Water surface elevation [m]"),
)

PANEL_ORDER = ("s", "speed", "d", "b", "h", "c", "t", "eta")


def _panels(stacks: dict[str, jnp.ndarray], frame: int) -> dict[str, np.ndarray]:
    u = stacks["u"][frame]
    v = stacks["v"][frame]
    return {
        "s": stacks["s"][frame],
        "speed": np.sqrt(u ** 2 + v ** 2),
        "d": stacks["d"][frame],
        "b": stacks["b"][frame],
        "h": stacks["h"][frame],
        "c": stacks["c"][frame],
        "t": stacks["t"][frame],
        "eta": stacks["s"][frame] + stacks["h"][frame] + stacks["b"][frame],
    }


def save_figure(stacks: dict[str, jnp.ndarray], run: RunConfig, frame: int = -1) -> str:
    os.makedirs(run.outdir, exist_ok=True)
    idx = frame if frame >= 0 else stacks["s"].shape[0] - 1
    data = _panels(stacks, idx)

    fig, ax = plt.subplots(2, 4, figsize=(16, 12))
    for i, (key, title) in enumerate(_TITLES):
        r, c = divmod(i, 4)
        im = ax[r, c].imshow(data[key], animated=True)
        ax[r, c].set_title(title, fontsize=_FS)
        ax[r, c].set_axis_off()
        fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04, orientation="horizontal")

    fig.suptitle(f"{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}", x=0.48, y=0.09, fontsize=_FS + 2)
    path = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}.png"
    fig.savefig(path, dpi=500)
    plt.close(fig)
    return path


def save_animation(stacks: dict[str, jnp.ndarray], run: RunConfig, fps: int = 10) -> str:
    os.makedirs(run.outdir, exist_ok=True)
    n_frames = stacks["s"].shape[0]
    init = _panels(stacks, 0)

    fig, ax = plt.subplots(2, 4, figsize=(16, 12))
    imgs = []
    for i, (key, title) in enumerate(_TITLES):
        r, c = divmod(i, 4)
        im = ax[r, c].imshow(init[key], animated=True)
        ax[r, c].set_title(title, fontsize=_FS)
        ax[r, c].set_axis_off()
        fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04, orientation="horizontal")
        imgs.append(im)

    title_obj = fig.suptitle(
        f"{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}", x=0.48, y=0.09, fontsize=_FS + 2
    )

    def updatefig(i):
        data = _panels(stacks, i)
        for im, key in zip(imgs, PANEL_ORDER):
            im.set_array(data[key])
        return imgs + [title_obj]

    ani = animation.FuncAnimation(fig, updatefig, interval=100, frames=n_frames, repeat=False, blit=True)
    Writer = animation.writers["ffmpeg"]
    writer = Writer(
        fps=fps,
        metadata=dict(artist="Roeland C. van de Vijsel & Johan van de Koppel"),
        bitrate=1800,
    )
    path = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}.mp4"
    ani.save(path, writer=writer)
    plt.close(fig)
    return path


def save_overview(stacks: dict[str, jnp.ndarray], run: RunConfig, facTY: float, frame_tides: float) -> str:
    """Mean/min/max over space for each variable, vs. time in years.

    `frame_tides` is the elapsed M2-tides between consecutive frames
    (EndTime / NumFrames); facTY converts tides to years.
    """
    os.makedirs(run.outdir, exist_ok=True)
    tt = np.arange(stacks["s"].shape[0] + 1)[1:] * facTY * frame_tides

    names = ("u", "v", "h", "b", "s", "d", "c", "t")
    fig, axs = plt.subplots(2, 4)
    for i, name in enumerate(names):
        r, c = divmod(i, 4)
        arr = stacks[name]
        axs[r, c].plot(tt, arr.mean(axis=(1, 2)), label=f"mean {name}")
        axs[r, c].plot(tt, arr.min(axis=(1, 2)), label=f"min {name}")
        axs[r, c].plot(tt, arr.max(axis=(1, 2)), label=f"max {name}")
        axs[r, c].set_title(f"{name} [{'m/s' if name in 'uv' else 'm' if name in 'hbs' else 'm^-2' if name == 'd' else 'm^(1/2)/s' if name == 'c' else 'N/m2'}]")
        axs[r, c].legend(fontsize=6)

    fig.suptitle(f"{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}", x=0.48, y=0.05)
    path = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}_overview.png"
    fig.savefig(path, dpi=500)
    plt.close(fig)
    return path