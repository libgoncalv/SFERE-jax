# The improvement layer: sfere.model.step with auditable, opt-in numerics
# changes (docs/numerics.md). The faithful port in sfere/model.py is never
# modified; with all options at their defaults this module delegates to the
# port itself, so "original" and "improvement" cannot silently mix.
#
# Each switch changes exactly one line-class of the original kernel and
# nothing else - same pressure gradient, same boundary conditions, same
# continuity/sediment/vegetation blocks, same diagnostics, same ordering:
#
#   semi_implicit_friction - kernel momentum friction -g_Ct2*|u|*u/h moves
#     from the explicit update into a backward-Euler divisor (1 + dT*lambda),
#     with lambda = g_Ct2*|u|/h using step-start |u| (the same lag the kernel
#     already uses). The update gain becomes 1/(1 + dT*lambda) <= 1 always:
#     the dT < 2/lambda cap (~0.017 s, set by vegetated thin films)
#     disappears, and dT becomes limited by the viscous term dX^2/(4*DifU),
#     which grows with dX.
#   upwind_advection - the momentum advection stencil d_dx/d_dy (symmetric,
#     centred) becomes the flow-direction (upwind) stencil. Centred + forward
#     Euler is unconditionally unstable for pure advection and survives only
#     while diffusion dominates the grid scale (grid Peclet number
#     Pe = |U|*dX/DifU << 1); the upwind stencil carries its own damping, so
#     larger dX stays usable. Momentum only: continuity/sediment/vegetation
#     keep the kernel's stencils.
#
# Guarantees tested in tests/test_numerics.py: the default path is the port
# itself; semi-implicit friction stays finite past the old dT cap where the
# explicit term diverges; the switched paths stay finite and actually differ.

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

from . import model, operators as ops
from .config import NumericsOptions
from .model import Params, _grad_row
from .state import State


def _upwind_x(z: jax.Array, w: jax.Array, dX: float) -> jax.Array:
    """w * dz/dx reading z from the cell the flow comes from (axis 1)."""
    return jnp.where(w >= 0.0,
                     (z - jnp.roll(z, 1, axis=1)) * w,
                     (jnp.roll(z, -1, axis=1) - z) * w) / dX


def _upwind_y(z: jax.Array, w: jax.Array, dY: float) -> jax.Array:
    """w * dz/dy reading z from the cell the flow comes from (axis 0)."""
    return jnp.where(w >= 0.0,
                     (z - jnp.roll(z, 1, axis=0)) * w,
                     (jnp.roll(z, -1, axis=0) - z) * w) / dY


def step_improved(state: State, p: Params, opt: NumericsOptions) -> State:
    """One simulation step with the improvement-layer numerics.

    All options off -> delegates to sfere.model.step (the faithful port).
    """
    if not (opt.semi_implicit_friction or opt.upwind_advection):
        return model.step(state, p)

    phys, grad = p.phys, p.gradients
    dT, dX, dY = p.dT, p.dX, p.dY
    H, W = state.h.shape

    # Per-column coefficient arrays (as in model.step)
    D0x = _grad_row(grad.D0, phys.D0, phys.D0min, phys.D0max, W)
    pDx = _grad_row(grad.pD, phys.pD, phys.pDmin, phys.pDmax, W)
    HinX = _grad_row(grad.Hin, phys.Hin, phys.HinMin, phys.HinMax, W)
    DifDx = _grad_row(grad.DifD, phys.DifD, phys.DifDmin, phys.DifDmax, W)

    # --- 1. Flow speed and wetting-drying clamp (unchanged) -------------------
    uabs = jnp.sqrt(state.u ** 2 + state.v ** 2)
    h = jnp.maximum(state.h, phys.HCrit)

    # --- 2. Friction rate (unchanged lagging, unchanged Ct) -------------------
    nn = phys.nb + (phys.nv - phys.nb) * (state.d / phys.kk)
    Ct = (1.0 / nn) * h ** (1.0 / 6.0)
    g_Ct2 = phys.g / (Ct * Ct)
    lam = g_Ct2 * uabs / h  # [1/s] the coefficient of the -lambda*u term

    # --- 3. Momentum: pressure + advection + diffusion; friction below --------
    if opt.upwind_advection:
        a_u = _upwind_x(state.u, state.u, dX) + _upwind_y(state.u, state.v, dY)
        a_v = _upwind_x(state.v, state.u, dX) + _upwind_y(state.v, state.v, dY)
    else:  # centred, as in the kernel
        a_u = state.u * ops.d_dx(state.u, dX) + state.v * ops.d_dy(state.u, dY)
        a_v = state.u * ops.d_dx(state.v, dX) + state.v * ops.d_dy(state.v, dY)
    du = (-phys.g * ops.dO_dx(h, state.s, state.b, dX)
          - a_u
          + phys.DifU * ops.d2_dxy2(state.u, dX, dY))
    dv = (-phys.g * ops.dO_dy(h, state.s, state.b, dY)
          - a_v
          + phys.DifU * ops.d2_dxy2(state.v, dX, dY))

    if opt.semi_implicit_friction:
        # backward Euler on the linear friction term: solve
        # (u - u_prev)/dT = du - lambda*u  for u  ->
        # u = (u_prev + dT*du) / (1 + dT*lambda); lambda lagged (step-start |u|)
        u = (state.u + dT * du) / (1.0 + dT * lam)
        v = (state.v + dT * dv) / (1.0 + dT * lam)
    else:
        u = state.u + dT * (du - g_Ct2 * uabs * state.u / h)
        v = state.v + dT * (dv - g_Ct2 * uabs * state.v / h)

    # --- Flow BCs (unchanged) ---------------------------------------------------
    u, v = ops.ReflectingBoundaries(u, v)
    u = ops.PersistentFluxBoundaries_bottom(u)
    v = ops.PersistentFluxBoundaries_bottom(v)

    # --- 4a. Continuity (unchanged, centred u*h fluxes) -------------------------
    dh = -ops.d_uh_dx(u, h, dX) - ops.d_vh_dy(v, h, dY)
    drh = HinX[jnp.newaxis, :] - phys.Df * h

    # --- 4b. Sediment (unchanged) ------------------------------------------------
    h_eff = h - phys.HCrit
    ds = (phys.S_in * h_eff / (phys.Qs + h_eff)
          - phys.E0 * (1.0 - phys.pE * state.d / phys.kk) * state.s * g_Ct2 * uabs ** 2
          + model._d2_dxy2_S(state.s, state.d, D0x, pDx, phys.kk, dX, dY))

    # --- 4c. Vegetation (unchanged) ----------------------------------------------
    dD = (phys.rr * state.d * (1.0 - state.d / phys.kk) * phys.Qq / (phys.Qq + h_eff)
          - phys.Ec * state.d * g_Ct2 * uabs ** 2
          + DifDx[jnp.newaxis, :] * ops.d2_dxy2(state.d, dX, dY))

    h = h + dT * (dh + drh)
    s = state.s + dT * ds
    d = state.d + dT * dD

    # --- 5. Scalar BCs (unchanged) -------------------------------------------------
    h = ops.NeumanBoundaries(h)
    d = ops.NeumanBoundaries(d)
    s = ops.NeumanBoundaries(s)
    s = s.at[H - 1, :].set(0.0)

    # --- 6. Diagnostics (unchanged, stale uabs/Ct) -----------------------------------
    c = Ct
    t = (phys.rho * phys.g / (Ct * Ct)) * uabs ** 2

    return State(u=u, v=v, h=h, b=state.b, s=s, d=d, c=c, t=t)
