# Improvement layer (sfere/numerics.py): delegation equality with the port,
# switch efficacy, and stability past the original friction-limited dT.

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfere import model
from sfere.config import GradientOptions, GridConfig, NumericsOptions, PhysParams
from sfere.model import Params
from sfere.numerics import step_improved
from sfere.state import init_conditions

_FIELDS = ("u", "v", "h", "b", "s", "d", "c", "t")


def _grid() -> GridConfig:
    return GridConfig(Block_Size_X=16, Block_Size_Y=16,
                      Block_Number_X=4, Block_Number_Y=4)


def _params(dT: float = 0.0125) -> Params:
    return Params(phys=PhysParams(), gradients=GradientOptions(),
                  dX=0.5, dY=0.5, dT=dT)


def _state():
    return init_conditions(PhysParams(), _grid(), vegetation=True, seed=2)


def _stiff_state():
    """Vegetated thin-film cells at the kernel's friction limit: h = HCrit,
    d = kk, |u| = 0.03 -> lambda = g*nv^2*|u|/h^(4/3) ~ 118 /s, so the
    explicit friction gain at dT = 0.05 is 1 - 5.9: amplifying."""
    state = _state()
    ones = np.full((64, 64), 1.0, dtype=np.float32)
    return state.replace(
        u=jnp.asarray(ones * 0.03),
        v=jnp.zeros((64, 64), dtype=np.float32),
        h=jnp.asarray(ones * 0.001),
        s=jnp.zeros((64, 64), dtype=np.float32),
        d=jnp.asarray(ones * 1500.0),
        c=jnp.zeros((64, 64), dtype=np.float32),
        t=jnp.zeros((64, 64), dtype=np.float32),
    )


def _make(step_fn, params, opt=None):
    """jit-compiled single-step closure over (params, opt)."""
    if opt is None:
        @jax.jit
        def f(state):
            return step_fn(state, params)
    else:
        @jax.jit
        def f(state):
            return step_fn(state, params, opt)
    return f


def test_default_options_delegate_to_the_port():
    state = _state()
    params = _params()
    reference = model.step(state, params)
    improved = step_improved(state, params, NumericsOptions())
    for name in _FIELDS:
        assert jnp.array_equal(getattr(reference, name), getattr(improved, name))


def test_explicit_friction_diverges_at_dt_0p05_but_semi_implicit_survives():
    params = _params(dT=0.05)  # beyond the kernel's dT < 2/lambda ~ 0.017 s
    explicit = _make(model.step, params)
    implicit = _make(step_improved, params,
                     NumericsOptions(semi_implicit_friction=True))

    exploded = False
    state = _stiff_state()
    for _ in range(300):
        state = explicit(state)
        if not np.isfinite(np.asarray(state.u)).all():
            exploded = True
            break
    assert exploded, "reference explicit friction should diverge at dT=0.05"

    state = _stiff_state()
    for _ in range(300):
        state = implicit(state)
    for name in _FIELDS:
        assert np.isfinite(np.asarray(getattr(state, name))).all()


def test_upwind_advection_differs_and_stays_finite():
    # The stencils only differ on a nonzero-gradient |u| field (uniform flow
    # is advected identically, to the last bit), so stamp in a shear pattern.
    def sheared():
        state = _state()
        u = np.asarray(state.u).copy()
        u[:, ::8] = 0.3
        return state.replace(u=jnp.asarray(u))

    reference, state = sheared(), sheared()
    assert not jnp.array_equal(
        np.asarray(model.step(reference, _params()).u),
        np.asarray(step_improved(state, _params(),
                                 NumericsOptions(upwind_advection=True)).u))

    stepper = _make(step_improved, _params(), NumericsOptions(upwind_advection=True))
    for _ in range(50):
        state = stepper(state)
        assert np.isfinite(np.asarray(state.u)).all()


def test_combined_switches_stay_finite_beyond_the_friction_cap():
    state = _stiff_state()
    stepper = _make(step_improved, _params(dT=0.05),
                    NumericsOptions(semi_implicit_friction=True,
                                    upwind_advection=True))
    for _ in range(300):
        state = stepper(state)
    for name in _FIELDS:
        assert np.isfinite(np.asarray(getattr(state, name))).all()
