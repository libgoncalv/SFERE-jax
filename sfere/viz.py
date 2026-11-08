# Visualization: 8-panel figure, MP4 animation, mean/min/max time series.
# Port of the plotting sections of the original notebook.
#
# The figure machinery is factored out so the notebook (sfere.live) can reuse
# it: draw_figure/set_frame build and update one figure object, and the
# save_* functions are thin wrappers over them.

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure

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

# Titles in the suptitle: spacing and unit format match the original notebook.
_SEP = "       -       "


def panel_values(u: np.ndarray, v: np.ndarray, h: np.ndarray, b: np.ndarray,
                 s: np.ndarray, d: np.ndarray, c: np.ndarray, t: np.ndarray
                 ) -> dict[str, np.ndarray]:
    """The 8 display panels from one frame's raw (2D) variables."""

    return {
        "s": s,
        "speed": np.sqrt(u ** 2 + v ** 2),
        "d": d,
        "b": b,
        "h": h,
        "c": c,
        "t": t,
        "eta": s + h + b,
    }


def _panels(stacks: dict[str, np.ndarray], frame: int) -> dict[str, np.ndarray]:
    return panel_values(
        u=stacks["u"][frame], v=stacks["v"][frame], h=stacks["h"][frame],
        b=stacks["b"][frame], s=stacks["s"][frame], d=stacks["d"][frame],
        c=stacks["c"][frame], t=stacks["t"][frame],
    )


def run_title(run: RunConfig) -> str:
    """Suptitle base, as in the original notebook."""
    return f"{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}"


def time_title(base: str, elapsed_tides: float, end_tides: float, facTY: float) -> str:
    """Suptitle with elapsed time in years, matching the original's updatefig."""
    return f"{base}{_SEP}Time: {elapsed_tides * facTY:.0f} of {end_tides * facTY:.0f} yr"


def draw_figure(data: dict[str, np.ndarray], title: str) -> tuple[Figure, dict]:
    """Build the 8-panel figure from one frame's panel data.

    Returns (fig, handles); handles["panels"] holds the 8 imshow artists in
    PANEL_ORDER and handles["title"] the suptitle text artist.
    """
    fig, ax = plt.subplots(2, 4, figsize=(16, 12))
    imgs = []
    for i, (key, panel_title) in enumerate(_TITLES):
        r, c = divmod(i, 4)
        im = ax[r, c].imshow(data[key], animated=True)
        ax[r, c].set_title(panel_title, fontsize=_FS)
        ax[r, c].set_axis_off()
        fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04, orientation="horizontal")
        imgs.append(im)

    title_obj = fig.suptitle(title, x=0.48, y=0.09, fontsize=_FS + 2)
    return fig, {"panels": imgs, "title": title_obj}


def set_frame(handles: dict, data: dict[str, np.ndarray], title: str | None = None) -> None:
    """Update the panel arrays (and optionally the suptitle) of a drawn figure."""
    for im, key in zip(handles["panels"], PANEL_ORDER):
        im.set_array(data[key])
    if title is not None:
        handles["title"].set_text(title)


def save_figure(stacks: dict[str, np.ndarray], run: RunConfig, frame: int = -1) -> str:
    os.makedirs(run.outdir, exist_ok=True)
    idx = frame if frame >= 0 else stacks["s"].shape[0] - 1
    data = _panels(stacks, idx)

    fig, _ = draw_figure(data, run_title(run))
    path = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}.png"
    fig.savefig(path, dpi=500)
    plt.close(fig)
    return path


def save_animation(stacks: dict[str, np.ndarray], run: RunConfig, fps: int = 10,
                   facTY: float | None = None, end_tides: float | None = None) -> str:
    """Save the stored frames as an MP4.

    When `facTY` and `end_tides` are given, each frame's suptitle carries the
    elapsed time in years (as in the original notebook); otherwise the title
    is static.
    """
    os.makedirs(run.outdir, exist_ok=True)
    n_frames = stacks["s"].shape[0]
    base = run_title(run)

    fig, handles = draw_figure(_panels(stacks, 0), base)

    def updatefig(i):
        title = time_title(base, (i + 1) / n_frames * end_tides, end_tides, facTY) \
            if facTY is not None and end_tides is not None else base
        set_frame(handles, _panels(stacks, i), title)
        return list(handles["panels"]) + [handles["title"]]

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


def save_overview(stacks: dict[str, np.ndarray], run: RunConfig, facTY: float, frame_tides: float) -> str:
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

    fig.suptitle(run_title(run), x=0.48, y=0.05)
    path = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}_overview.png"
    fig.savefig(path, dpi=500)
    plt.close(fig)
    return path
