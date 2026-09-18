# Frame-level simulation loop: lax.fori_loop inside jit, frames in Python.

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

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
    """Run the full simulation, returning (states, per-frame wall times).

    Mirrors the original: the whole time series stays in device memory
    (NumFrames snapshots). For the RUN9 default that is
    100 frames x 8 vars x 2048x1024 float32 ~= 6.7 GB; pass a reduced
    NumFrames for constrained GPUs.
    """
    steps_per_frame = sim.steps_per_frame
    frame = make_stepped(params, steps_per_frame)

    states = []
    times = []
    for count in range(sim.NumFrames):
        t0 = time.time()
        state = frame(state)
        jax.block_until_ready(state.h)
        states.append(state)
        times.append(time.time() - t0)
        if verbose:
            print(f"frame {count + 1}/{sim.NumFrames}: {times[-1]:.2f} s", flush=True)

    return states, times


def stack(states: list[State]) -> dict[str, jax.Array]:
    """Stack frame list into (Frame, H, W) arrays keyed by variable name."""
    return {
        name: jnp.stack([getattr(st, name) for st in states], axis=0)
        for name in ("u", "v", "h", "b", "s", "d", "c", "t")
    }