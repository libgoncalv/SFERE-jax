# Frame-level simulation loop: lax.fori_loop inside jit, frames in Python.

from __future__ import annotations

import time

import jax
import numpy as np

from .config import SimConfig
from .model import Params, step
from .state import State


def make_stepped(params: Params, steps_per_frame: int):
    """Return a jit-compiled function running `steps_per_frame` model steps."""

    @jax.jit
    def frame(state: State) -> State:
        def body(_, st):
            return step(st, params)

        return jax.lax.fori_loop(0, steps_per_frame, body, state)

    return frame


def run_frames(state: State, params: Params, sim: SimConfig, verbose: bool = True):
    """Run the full simulation, returning (frames, per-frame wall times).

    Each frame is copied to host numpy immediately (mirroring the original's
    cl.enqueue_copy pattern), so device memory stays at one working state and
    host memory holds ~6.7 GB for the RUN9 default. Returns a list of
    variable-keyed numpy dicts, one per frame.
    """
    steps_per_frame = sim.steps_per_frame
    frame = make_stepped(params, steps_per_frame)

    var_names = ("u", "v", "h", "b", "s", "d", "c", "t")
    frames = []
    times = []
    for count in range(sim.NumFrames):
        t0 = time.time()
        state = frame(state)
        jax.block_until_ready(state.h)
        moved = {name: np.asarray(getattr(state, name)) for name in var_names}
        frames.append(moved)
        times.append(time.time() - t0)
        if verbose:
            print(f"frame {count + 1}/{sim.NumFrames}: {times[-1]:.2f} s", flush=True)

    return frames, times


def stack(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-frame dicts into (Frame, H, W) arrays keyed by variable name."""
    return {name: np.stack([f[name] for f in frames], axis=0) for name in frames[0]}