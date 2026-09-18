# Matrix stencil operators and boundary conditions (port of HydroFunctions_iPy.cl).
#
# All functions operate on (H, W) arrays. Index convention matches the original
# OpenCL code: row 0 = top, column 0 = left; original flat index = row*W + col.
# Stencils use jnp.roll ghosts; boundary-cell results are always discarded
# because every edge is subsequently overwritten by a boundary-condition call
# (the original kernel computes stencils on interior threads only).
#
# Boundary functions replicate the original's if/else-if precedence: the top
# row branch runs first, so corners take the top-row formula; bottom corners
# take the bottom-row formula; side columns apply to interior rows only.

from __future__ import annotations

import jax
import jax.numpy as jnp


def d2_dxy2(z: jax.Array, dX: float, dY: float) -> jax.Array:
    """Laplacian: (z[left]+z[right]-2z)/dX^2 + (z[top]+z[bottom]-2z)/dY^2."""
    left = jnp.roll(z, 1, axis=1)
    right = jnp.roll(z, -1, axis=1)
    top = jnp.roll(z, 1, axis=0)
    bottom = jnp.roll(z, -1, axis=0)
    return (left + right - 2.0 * z) / dX / dX + (top + bottom - 2.0 * z) / dY / dY


def d_dx(z: jax.Array, dX: float) -> jax.Array:
    """Central d/dx: (z[right]-z[left])/2/dX."""
    return (jnp.roll(z, -1, axis=1) - jnp.roll(z, 1, axis=1)) / 2.0 / dX


def d_dy(z: jax.Array, dY: float) -> jax.Array:
    """Central d/dy: (z[bottom]-z[top])/2/dY."""
    return (jnp.roll(z, -1, axis=0) - jnp.roll(z, 1, axis=0)) / 2.0 / dY


def dO_dx(h: jax.Array, s: jax.Array, b: jax.Array, dX: float) -> jax.Array:
    """Pressure gradient in x: d(h+s+b)/dx."""
    z = h + s + b
    return (jnp.roll(z, -1, axis=1) - jnp.roll(z, 1, axis=1)) / 2.0 / dX


def dO_dy(h: jax.Array, s: jax.Array, b: jax.Array, dY: float) -> jax.Array:
    """Pressure gradient in y: d(h+s+b)/dy."""
    z = h + s + b
    return (jnp.roll(z, -1, axis=0) - jnp.roll(z, 1, axis=0)) / 2.0 / dY


def d_uh_dx(u: jax.Array, h: jax.Array, dX: float) -> jax.Array:
    """(u*h) gradient in x: (u[right]*h[right]-u[left]*h[left])/2/dX."""
    uh = u * h
    return (jnp.roll(uh, -1, axis=1) - jnp.roll(uh, 1, axis=1)) / 2.0 / dX


def d_vh_dy(v: jax.Array, h: jax.Array, dY: float) -> jax.Array:
    """(v*h) gradient in y: (v[bottom]*h[bottom]-v[top]*h[top])/2/dY."""
    vh = v * h
    return (jnp.roll(vh, -1, axis=0) - jnp.roll(vh, 1, axis=0)) / 2.0 / dY


# ---------------------------------------------------------------------------
# Boundary conditions. Each sets values on the four edge rows/columns with the
# same formulas and branch precedence as HydroFunctions_iPy.cl.
# ---------------------------------------------------------------------------

def PeriodicBoundaries(z: jax.Array) -> jax.Array:
    """Top <- row H-2, bottom <- row 1, left <- col W-2, right <- col 1."""
    H, W = z.shape
    return (
        z.at[0, :].set(z[H - 2, :])
        .at[H - 1, :].set(z[1, :])
        .at[1:H - 1, 0].set(z[1:H - 1, W - 2])
        .at[1:H - 1, W - 1].set(z[1:H - 1, 1])
    )


def NeumanBoundaries(z: jax.Array) -> jax.Array:
    """Zero-flux: copy the adjacent interior value onto each edge."""
    H, W = z.shape
    return (
        z.at[0, :].set(z[1, :])
        .at[H - 1, :].set(z[H - 2, :])
        .at[1:H - 1, 0].set(z[1:H - 1, 1])
        .at[1:H - 1, W - 1].set(z[1:H - 1, W - 2])
    )


def ReflectingBoundaries(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solid wall: u flips sign across vertical walls, v across horizontal walls."""
    H, W = u.shape
    u = (
        u.at[0, :].set(u[1, :])
        .at[1:H - 1, 0].set(-u[1:H - 1, 1])
        .at[1:H - 1, W - 1].set(-u[1:H - 1, W - 2])
    )
    v = (
        v.at[0, :].set(-v[1, :])
        .at[1:H - 1, 0].set(v[1:H - 1, 1])
        .at[1:H - 1, W - 1].set(v[1:H - 1, W - 2])
    )
    return u, v


def PersistentFluxBoundaries(z: jax.Array) -> jax.Array:
    """Linear extrapolation over each edge: z[edge] = 2*z[interior1] - z[interior2]."""
    H, W = z.shape
    return (
        z.at[0, :].set(2.0 * z[1, :] - z[2, :])
        .at[H - 1, :].set(2.0 * z[H - 2, :] - z[H - 3, :])
        .at[1:H - 1, 0].set(2.0 * z[1:H - 1, 1] - z[1:H - 1, 2])
        .at[1:H - 1, W - 1].set(2.0 * z[1:H - 1, W - 2] - z[1:H - 1, W - 3])
    )


def PersistentFluxBoundaries_bottom(z: jax.Array) -> jax.Array:
    """Bottom-row-only variant: what the kernel's bottom-boundary threads execute."""
    return z.at[-1, :].set(2.0 * z[-2, :] - z[-3, :])


def DirichletBoundaries(z: jax.Array, Value: float) -> jax.Array:
    """Fixed value on all four edges."""
    return z.at[0, :].set(Value).at[-1, :].set(Value).at[:, 0].set(Value).at[:, -1].set(Value)