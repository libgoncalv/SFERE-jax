# One simulation step: direct port of SimulationKernel from
# clPy.ComplexChannelPatterns_Fig1f_RUN9.ipynb.
#
# Ordering preserved from the original kernel:
#   1. uabs from pre-update flow; clamp h >= HCrit everywhere (before momentum)
#   2. nn/Ct from clamped h and pre-update vegetation d
#   3. momentum (interior only), then flow BCs:
#      top row + side columns = reflecting (corners follow the kernel's
#      branch precedence), bottom row = persistent flux (free outflow)
#   4. continuity / sediment / vegetation (interior only, post-momentum u,v,
#      clamped h, pre-update s and d)
#   5. scalar BCs: h and d Neumann on all edges; s Neumann except bottom row,
#      which is Dirichlet-0
#   6. diagnostics c, t on all cells, from the STALE pre-update uabs and Ct
#      (the original recomputes them at kernel end from step-start values)
#
# Differences vs. the OpenCL original (deliberate, see plan): derivatives are
# evaluated sequentially from consistent pre-update fields instead of racing
# mid-kernel writes; boundary logic is applied via disjoint .at[].set() calls.

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import operators as ops
from .config import GradientOptions, PhysParams
from .state import State


@dataclass(frozen=True)
class Params:
    """Bundle of everything the step needs; treated as compile-time constants."""

    phys: PhysParams
    gradients: GradientOptions
    dX: float   # [m] cell width
    dY: float   # [m] cell height
    dT: float   # [s] time step


def _grad_row(on: bool, x: float, xmin: float, xmax: float, W: int) -> jax.Array:
    """Per-column values for a switchable across-width logarithmic gradient.

    Matches the kernel's `exp(log(max)-log(min))*column/W + log(min)` formulas
    (the multiplication binds before the addition of log(min), inside exp).
    """
    if on:
        frac = jnp.arange(W, dtype=jnp.float32) / jnp.float32(W)
        return jnp.exp((jnp.log(xmax) - jnp.log(xmin)) * frac + jnp.log(xmin))
    return jnp.full((W,), x, dtype=jnp.float32)


def step(state: State, p: Params) -> State:
    phys, grad = p.phys, p.gradients
    dT, dX, dY = p.dT, p.dX, p.dY
    H, W = state.h.shape

    # Per-column coefficient arrays (constants when the gradient switch is off)
    D0x = _grad_row(grad.D0, phys.D0, phys.D0min, phys.D0max, W)
    pDx = _grad_row(grad.pD, phys.pD, phys.pDmin, phys.pDmax, W)
    HinX = _grad_row(grad.Hin, phys.Hin, phys.HinMin, phys.HinMax, W)
    DifDx = _grad_row(grad.DifD, phys.DifD, phys.DifDmin, phys.DifDmax, W)

    # --- 1. Flow speed and wetting-drying clamp -------------------------------
    uabs = jnp.sqrt(state.u ** 2 + state.v ** 2)
    h = jnp.maximum(state.h, phys.HCrit)

    # --- 2. Friction ------------------------------------------------------------
    nn = phys.nb + (phys.nv - phys.nb) * (state.d / phys.kk)
    Ct = (1.0 / nn) * h ** (1.0 / 6.0)
    g_Ct2 = phys.g / (Ct * Ct)

    # --- 3. Momentum (interior values used; edges overwritten by BCs below) -----
    du = (-phys.g * ops.dO_dx(h, state.s, state.b, dX)
          - state.u * ops.d_dx(state.u, dX)
          - state.v * ops.d_dy(state.u, dY)
          - g_Ct2 * uabs * state.u / h
          + phys.DifU * ops.d2_dxy2(state.u, dX, dY))
    dv = (-phys.g * ops.dO_dy(h, state.s, state.b, dY)
          - state.u * ops.d_dx(state.v, dX)
          - state.v * ops.d_dy(state.v, dY)
          - g_Ct2 * uabs * state.v / h
          + phys.DifU * ops.d2_dxy2(state.v, dX, dY))

    u = state.u + dT * du
    v = state.v + dT * dv

    # Flow BCs (disjoint writes): top+left+right reflecting, bottom persistent flux
    u, v = ops.ReflectingBoundaries(u, v)
    u = ops.PersistentFluxBoundaries_bottom(u)
    v = ops.PersistentFluxBoundaries_bottom(v)

    # --- 4a. Continuity (post-momentum u,v; clamped h) ---------------------------
    dh = -ops.d_uh_dx(u, h, dX) - ops.d_vh_dy(v, h, dY)
    drh = HinX[jnp.newaxis, :] - phys.Df * h

    # --- 4b. Sediment -------------------------------------------------------------
    h_eff = h - phys.HCrit
    ds = (phys.S_in * h_eff / (phys.Qs + h_eff)
          - phys.E0 * (1.0 - phys.pE * state.d / phys.kk) * state.s * g_Ct2 * uabs ** 2
          + _d2_dxy2_S(state.s, state.d, D0x, pDx, phys.kk, dX, dY))

    # --- 4c. Vegetation -------------------------------------------------------------
    dD = (phys.rr * state.d * (1.0 - state.d / phys.kk) * phys.Qq / (phys.Qq + h_eff)
          - phys.Ec * state.d * g_Ct2 * uabs ** 2
          + DifDx[jnp.newaxis, :] * ops.d2_dxy2(state.d, dX, dY))

    h = h + dT * (dh + drh)
    s = state.s + dT * ds
    d = state.d + dT * dD

    # --- 5. Scalar BCs ---------------------------------------------------------------
    h = ops.NeumanBoundaries(h)
    d = ops.NeumanBoundaries(d)
    s = ops.NeumanBoundaries(s)
    s = s.at[H - 1, :].set(0.0)  # Dirichlet-0 on the bottom (outflow) edge

    # --- 6. Diagnostics (stale, all cells) ---------------------------------------------
    c = Ct
    t = (phys.rho * phys.g / (Ct * Ct)) * uabs ** 2

    return State(u=u, v=v, h=h, b=state.b, s=s, d=d, c=c, t=t)


def _d2_dxy2_S(s: jax.Array, d: jax.Array, D0x: jax.Array, pDx: jax.Array,
               kk: float, dX: float, dY: float) -> jax.Array:
    """Vegetation-modulated sediment diffusion, exact port of d2_dxy2_S.

    For interior cell (row, col) the kernel computes
        difco(d[right], d[col], col)*(s[right]-s[col])/dx^2
      - difco(d[col], d[left],  col)*(s[col]-s[left])/dx^2
      + difco(d[bottom], d[col], col)*(s[bottom]-s[col])/dy^2
      - difco(d[col], d[top],   col)*(s[col]-s[top])/dy^2
    with difco(d1, d2, col) = D0x[col]*(1 - pDx[col]*((d1+d2)/2)/kk).
    Note both x-terms use the CURRENT column's D0x/pDx (as in the kernel), so
    this is deliberately not the symmetric conservative form when gradients
    are enabled.
    """
    dC = d[1:-1, 1:-1]                     # current-cell biomass, interior
    D0i = D0x[1:-1][jnp.newaxis, :]        # current-column coefficients
    pDi = pDx[1:-1][jnp.newaxis, :]

    def difco(dm: jax.Array) -> jax.Array:
        return D0i * (1.0 - pDi * dm / kk)

    dm_e = 0.5 * (d[1:-1, 2:] + dC)        # between right neighbour and current
    dm_w = 0.5 * (dC + d[1:-1, :-2])       # between current and left neighbour
    dm_s = 0.5 * (d[2:, 1:-1] + dC)        # bottom
    dm_n = 0.5 * (dC + d[:-2, 1:-1])       # top

    x_term = (difco(dm_e) * (s[1:-1, 2:] - s[1:-1, 1:-1])
              - difco(dm_w) * (s[1:-1, 1:-1] - s[1:-1, :-2])) / dX / dX
    y_term = (difco(dm_s) * (s[2:, 1:-1] - s[1:-1, 1:-1])
              - difco(dm_n) * (s[1:-1, 1:-1] - s[:-2, 1:-1])) / dY / dY

    out = jnp.zeros_like(s)
    return out.at[1:-1, 1:-1].set(x_term + y_term)