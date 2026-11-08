# Frame-level simulation loop: lax.fori_loop inside jit, frames in Python.

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax
import numpy as np

from .config import SimConfig
from .model import Params, step
from .state import State

if TYPE_CHECKING:
    from .config import RunConfig as _RunConfig

_VAR_NAMES = ("u", "v", "h", "b", "s", "d", "c", "t")


def make_stepped(params: Params, steps_per_frame: int):
    """Return a jit-compiled function running `steps_per_frame` model steps."""

    @jax.jit
    def frame(state: State) -> State:
        def body(_, st):
            return step(st, params)

        return jax.lax.fori_loop(0, steps_per_frame, body, state)

    return frame


class Simulation:
    """Incremental, resumable simulation driver shared by the CLI and notebooks.

    Holds one jit-compiled frame stepper (one per SimConfig/Params, so a
    continued run never re-jits) and the on-device State; every advance()
    captures the frame on host immediately (mirroring the original's
    cl.enqueue_copy pattern), so device memory stays at one working state.
    """

    def __init__(self, state: State, params: Params, sim: SimConfig) -> None:
        self.params = params
        self.sim = sim
        self._frame = make_stepped(params, sim.steps_per_frame)
        self._state = state
        self.frames: list[dict[str, np.ndarray]] = []
        self.times: list[float] = []

    # -- state introspection --------------------------------------------------

    @property
    def state(self) -> State:
        """The current State on the device (after all advanced frames)."""
        return self._state

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def tides_per_frame(self) -> float:
        """Simulated M2-tides captured per frame."""
        return self.sim.steps_per_frame * self.sim.dT

    @property
    def tides_elapsed(self) -> float:
        """Total simulated M2-tides so far."""
        return self.n_frames * self.tides_per_frame

    def latest_frame(self) -> dict[str, np.ndarray]:
        """The most recent frame as host numpy; the initial state if none."""

        if not self.frames:
            return {name: np.asarray(getattr(self._state, name)) for name in _VAR_NAMES}
        return self.frames[-1]

    # -- driving ----------------------------------------------------------------

    def advance(self, n_frames: int = 1) -> dict[str, np.ndarray]:
        """Advance `n_frames`, capturing each on host; returns the latest frame."""

        for _ in range(n_frames):
            t0 = time.time()
            self._state = self._frame(self._state)
            jax.block_until_ready(self._state.h)
            moved = {name: np.asarray(getattr(self._state, name)) for name in _VAR_NAMES}
            self.frames.append(moved)
            self.times.append(time.time() - t0)
        return self.frames[-1]

    # -- output -------------------------------------------------------------------

    def save(self, run: "_RunConfig", *, npy: bool = True, figure: bool = True,
             animation: bool = True, overview: bool = True) -> dict[str, str]:
        """Save the captured frames with the original notebook's filenames.

        Returns a dict of written paths; keys present only for the parts enabled.
        io/viz are imported lazily (io imports simulate, so a top-level import
        would be circular).
        """

        if not self.frames:
            raise ValueError("no frames captured; call advance() first")
        from .io import save_all
        from .viz import save_animation, save_figure, save_overview

        written: dict[str, str] = {}
        stacks = stack(self.frames)
        if npy:
            save_all(stacks, run)
        if figure:
            written["figure"] = save_figure(stacks, run)
        if animation:
            written["animation"] = save_animation(stacks, run)
        if overview:
            written["overview"] = save_overview(stacks, run, self.params.phys.facTY,
                                                frame_tides=self.tides_per_frame)
        return written


def run_frames(state: State, params: Params, sim: SimConfig, verbose: bool = True):
    """Run the full simulation, returning (frames, per-frame wall times).

    Thin wrapper over Simulation kept for the CLI: output (frames format and
    progress lines) is identical to the pre-Simulation implementation.
    """
    nsim = Simulation(state, params, sim)
    for count in range(sim.NumFrames):
        nsim.advance()
        if verbose:
            print(f"frame {count + 1}/{sim.NumFrames}: {nsim.times[-1]:.2f} s", flush=True)

    return nsim.frames, nsim.times


def stack(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-frame dicts into (Frame, H, W) arrays keyed by variable name."""
    return {name: np.stack([f[name] for f in frames], axis=0) for name in frames[0]}
