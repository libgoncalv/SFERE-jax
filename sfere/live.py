# Notebook display helpers: a live-updating 8-panel figure and inline
# animations, rendered through explicit IPython.display calls so the Agg
# backend (set by sfere.viz) keeps working in any notebook client.
#
# IPython is imported lazily inside the methods: nothing here is needed
# outside a notebook, and tests/CLI stay importable without it.

from __future__ import annotations

import io
from typing import Any

import numpy as np

from . import viz
from .config import RunConfig
from .simulate import Simulation


def _ipython():
    """Import IPython.display lazily; notebooks only."""
    try:
        from IPython import display as D
    except ImportError as exc:
        raise ImportError("sfere.live needs IPython to display in a notebook"
                          " (pip install ipython, included in requirements.txt)") from exc
    return D


def _panel_data(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """Panel data for one frame (a host dict of the 8 raw variables)."""
    return viz.panel_values(
        u=frame["u"], v=frame["v"], h=frame["h"], b=frame["b"],
        s=frame["s"], d=frame["d"], c=frame["c"], t=frame["t"],
    )


class LiveFigure:
    """The 8-panel figure shown once, then updated in place in the notebook.

    LiveFigure(simulation, run): the Simulation supplies frames, time
    bookkeeping and the tides-to-years conversion for the suptitle; `run`
    (optional) labels the run as in the saved figure. Every show() re-renders
    the same figure object (panel arrays + elapsed-time title) as a PNG and
    updates a single IPython display, so cells add no figure spam.
    """

    def __init__(self, sim: Simulation, run: RunConfig | None = None) -> None:
        self._sim = sim
        self._base = viz.run_title(run) if run is not None else ""
        self._fig = None
        self._handles: dict | None = None
        self._handle = None  # IPython DisplayHandle, set on first push

    def show(self, frame: dict[str, np.ndarray] | None = None) -> None:
        """Render (first call) or update the figure; `frame` defaults to the
        simulation's latest frame."""

        if frame is None:
            frame = self._sim.latest_frame()
        data = _panel_data(frame)

        title = self._title()
        if self._handles is None:
            self._fig, self._handles = viz.draw_figure(data, title)
        else:
            viz.set_frame(self._handles, data, title or None)
        self._push()

    def _title(self) -> str:
        """Suptitle with elapsed time, as in the original's per-frame titles."""
        if not self._base:
            return ""
        elapsed = self._sim.tides_elapsed
        end_tides = max(self._sim.sim.EndTime, elapsed)
        return viz.time_title(self._base, elapsed, end_tides, self._sim.params.phys.facTY)

    def _push(self) -> None:
        """Rasterize the current canvas and display/update it via IPython.

        An explicit DisplayHandle with our own id is used instead of
        display(display_id=True): it returns a usable handle even outside a
        live IPython shell, where display() would return None.
        """
        D = _ipython()

        buf = io.BytesIO()
        self._fig.savefig(buf, format="png")
        img = D.Image(data=buf.getvalue(), format="png")
        if self._handle is None:
            self._handle = D.DisplayHandle()
            D.display(img, display_id=self._handle.display_id)
        else:
            self._handle.update(img)


def html_animation(sim: Simulation, run: RunConfig | None = None, fps: int = 10):
    """An inline (HTML/JS) 8-panel animation of every frame captured so far.

    The suptitle shows elapsed time per frame, as in the original notebook's
    FuncAnimation. Returns IPython.display.HTML; a notebook cell returning it
    renders it inline. Renders directly from the per-frame dicts (no stack
    copy), so memory stays at what the frames already hold.
    """
    if not sim.frames:
        raise ValueError("no frames captured; call advance() first")
    D = _ipython()
    from matplotlib import animation
    import matplotlib.pyplot as plt

    base = viz.run_title(run) if run is not None else ""
    n = len(sim.frames)
    end_tides = max(sim.sim.EndTime, n * sim.tides_per_frame)

    def title(i: int) -> str:
        return viz.time_title(base, (i + 1) * sim.tides_per_frame, end_tides, sim.params.phys.facTY)

    fig, handles = viz.draw_figure(_panel_data(sim.frames[0]), title(0))

    def update(i):
        viz.set_frame(handles, _panel_data(sim.frames[i]), title(i))
        return []

    ani = animation.FuncAnimation(fig, update, interval=100, frames=n, repeat=False, blit=False)
    html = ani.to_jshtml(fps=fps)
    plt.close(fig)
    return D.HTML(html)
