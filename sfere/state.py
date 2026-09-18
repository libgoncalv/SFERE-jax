# State container and initial conditions.

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from .config import GridConfig, PhysParams

_FIELDS = ("u", "v", "h", "b", "s", "d", "c", "t")


@dataclasses.dataclass(frozen=True)
class State:
    """The 8 model variables as (Grid_Height, Grid_Width) float32 arrays.

    Axis 0 = original `row` (code-Y, top to bottom); axis 1 = original
    `column` (code-X, left to right). See README for the code-vs-manuscript
    coordinate flip.
    """

    u: jax.Array  # [m/s] flow in code-X direction
    v: jax.Array  # [m/s] flow in code-Y direction
    h: jax.Array  # [m] water layer thickness
    b: jax.Array  # [m] subsediment bed topography
    s: jax.Array  # [m] sediment layer thickness
    d: jax.Array  # [m^-2] vegetation stem density
    c: jax.Array  # [m^(1/2)/s] Chezy coefficient (diagnostic)
    t: jax.Array  # [N/m2] bed shear stress (diagnostic)

    def replace(self, **kw) -> "State":
        return dataclasses.replace(self, **kw)


# Register after class definition (register_dataclass returns the type).
jax.tree_util.register_dataclass(State, data_fields=_FIELDS, meta_fields=())


def init_conditions(phys: PhysParams, grid: GridConfig, vegetation: bool = True, seed: int = 2) -> State:
    """Uniform initial conditions, mirroring the original notebook.

    Note: the original seeds with np.random.seed(2) and draws vegetation
    seedlings from the global numpy RNG; we reproduce that exactly.
    """

    H, W = grid.Grid_Height, grid.Grid_Width

    h_uni = phys.H0
    v_uni = (np.sqrt(phys.slope) / phys.nb) * h_uni ** (2 / 3)
    if v_uni == 0.0:
        S_uni = 0.0
    else:
        S_uni = (phys.S_in / phys.E0) * ((h_uni - phys.HCrit) * h_uni ** (1 / 3) / (phys.Qs + h_uni - phys.HCrit)) / (
            phys.g * phys.nb * phys.nb * v_uni * v_uni
        )

    # X left-to-right 0..1, Y top-to-bottom 0..1 (matches np.meshgrid usage)
    _X, Y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))

    np.random.seed(seed)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.full((H, W), v_uni, dtype=np.float32)
    h = np.full((H, W), h_uni, dtype=np.float32)
    b = ((1.0 - Y) * phys.slope * grid.LengthY).astype(np.float32)
    s = np.full((H, W), S_uni, dtype=np.float32)
    if vegetation:
        d = (np.random.rand(H * W) < phys.p_est).astype(np.float32).reshape(H, W) * phys.kk
    else:
        d = np.zeros((H, W), dtype=np.float32)
    c = np.full((H, W), phys.Cb, dtype=np.float32)
    t = np.full((H, W), (phys.rho * phys.g / phys.Cb / phys.Cb) * v_uni * v_uni, dtype=np.float32)

    return State(
        u=jnp.asarray(u), v=jnp.asarray(v), h=jnp.asarray(h), b=jnp.asarray(b),
        s=jnp.asarray(s), d=jnp.asarray(d), c=jnp.asarray(c), t=jnp.asarray(t),
    )