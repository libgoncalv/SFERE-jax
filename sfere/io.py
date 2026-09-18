# Output saving with filenames identical to the original notebook.

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from .config import RunConfig
from .simulate import stack


def save_all(stacks: dict[str, jnp.ndarray], run: RunConfig) -> None:
    """Save final state and full stacks as .npy, mirroring the original saves.

    Original final-state names end in _u.._d; stack names in _us.._ds. b and c
    stacks are static/diagnostic in the original full-save list (b is constant;
    the original saves _bs too) - we save the same set it did: us,vs,hs,bs,ss,ds.
    """
    os.makedirs(run.outdir, exist_ok=True)
    base = f"{run.outdir}/{run.ModelName}_SUBRUNa{run.SUBRUNa}_SUBRUNb{run.SUBRUNb}"

    for name in ("u", "v", "h", "b", "s", "d"):
        np.save(f"{base}_{name}", np.asarray(stacks[name][-1]))

    for name in ("u", "v", "h", "b", "s", "d"):
        np.save(f"{base}_{name}s", np.asarray(stacks[name]))


def last_frame_dict(stacks: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {k: v[-1] for k, v in stacks.items()}