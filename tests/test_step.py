# Step-level physics tests: invariants, ordering, uniform-stability.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfere.config import GradientOptions, GridConfig, PhysParams
from sfere.model import Params, step
from sfere.state import State, init_conditions

GRID = GridConfig(Block_Size_X=16, Block_Size_Y=16, Block_Number_X=2, Block_Number_Y=2)
PARAMS = Params(phys=PhysParams(), gradients=GradientOptions(), dX=GRID.dX, dY=GRID.dY, dT=0.0125)


def small_state(**overrides):
    st = init_conditions(PhysParams(), GRID, vegetation=True, seed=2)
    kw = {f: getattr(st, f) for f in ("u", "v", "h", "b", "s", "d", "c", "t")}
    kw.update(overrides)
    return State(**kw)


def test_state_is_pytree():
    st = small_state()
    leaves = jax.tree.leaves(st)
    assert len(leaves) == 8
    flat = jax.tree.flatten_with_path(st)
    assert len(flat[0]) == 8


def test_no_nan_after_steps():
    st = small_state()
    for _ in range(50):
        st = step(st, PARAMS)
    for name in ("u", "v", "h", "b", "s", "d", "c", "t"):
        arr = np.asarray(getattr(st, name))
        assert np.isfinite(arr).all(), name


def test_h_clamped_everywhere():
    st = small_state(h=jnp.full((GRID.Grid_Height, GRID.Grid_Width), -0.5, dtype=jnp.float32))
    st = step(st, PARAMS)
    assert float(np.asarray(st.h).min()) >= PhysParams().HCrit - 1e-6


def test_vegetation_bounded():
    st = small_state()
    for _ in range(20):
        st = step(st, PARAMS)
    d = np.asarray(st.d)
    kk = PhysParams().kk
    assert d.min() >= -1e-3
    assert d.max() <= kk + 1e-3


def test_sediment_nonnegative():
    st = small_state()
    for _ in range(20):
        st = step(st, PARAMS)
    assert float(np.asarray(st.s).min()) >= -1e-4


def test_sediment_dirichlet_zero_bottom():
    st = small_state()
    st = step(st, PARAMS)
    assert np.allclose(np.asarray(st.s)[-1, :], 0.0)


def test_bed_topography_static():
    st0 = small_state()
    st1 = step(st0, PARAMS)
    assert np.array_equal(np.asarray(st0.b), np.asarray(st1.b))


def test_diagnostics_match_stale_values():
    """c, t at end of step must equal Ct/uabs computed from step-start u,v,h,d
    (after the HCrit clamp), matching the original kernel's stale diagnostics."""
    st0 = small_state()
    phys = PhysParams()
    h = jnp.maximum(st0.h, phys.HCrit)
    nn = phys.nb + (phys.nv - phys.nb) * (st0.d / phys.kk)
    Ct = (1.0 / nn) * h ** (1.0 / 6.0)
    uabs = jnp.sqrt(st0.u ** 2 + st0.v ** 2)
    st1 = step(st0, PARAMS)
    assert np.allclose(np.asarray(st1.c), np.asarray(Ct), rtol=1e-5)
    expected_t = (phys.rho * phys.g / (Ct * Ct)) * uabs ** 2
    assert np.allclose(np.asarray(st1.t), np.asarray(expected_t), rtol=1e-5)


def test_uniform_state_is_stationary():
    """A flat, vegetation-free field with zero slope must stay uniform
    (up to the boundary rows) — a basic stability smoke test."""
    phys = PhysParams()
    st = init_conditions(phys, GRID, vegetation=False, seed=0)
    for _ in range(100):
        st = step(st, PARAMS)
    interior = np.asarray(st.h)[1:-1, 1:-1]
    assert np.allclose(interior, phys.H0, atol=1e-4), f"h drifted: {interior.min()}..{interior.max()}"


def test_jit_step_matches_eager():
    st0 = small_state()
    jitted = jax.jit(lambda s: step(s, PARAMS))
    a = step(st0, PARAMS)
    b = jitted(st0)
    for name in ("u", "v", "h", "s", "d"):
        assert np.allclose(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), rtol=1e-5)