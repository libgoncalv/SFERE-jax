# Tests for matrix stencil operators (vs. hand-computed finite differences).

import jax.numpy as jnp
import numpy as np
import pytest

from sfere.operators import d2_dxy2, d_dx, d_dy, d_uh_dx, d_vh_dy

D = 0.5  # dX == dY == 0.5


@pytest.fixture
def grid_2d():
    """4x5 grid with distinct values so every stencil index is distinguishable."""
    return jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 8.0


def test_d_dx_interior(grid_2d):
    z = grid_2d
    got = d_dx(z, D)
    # Interior columns: (z[:, j+1]-z[:, j-1])/(2*D); check a few by hand.
    assert np.allclose(got[2, 2], (z[2, 3] - z[2, 1]) / (2 * D), atol=1e-6)
    assert np.allclose(got[0, 3], (z[0, 4] - z[0, 2]) / (2 * D), atol=1e-6)


def test_d_dy_interior(grid_2d):
    z = grid_2d
    got = d_dy(z, D)
    assert np.allclose(got[1, 1], (z[2, 1] - z[0, 1]) / (2 * D), atol=1e-6)
    assert np.allclose(got[2, 3], (z[3, 3] - z[1, 3]) / (2 * D), atol=1e-6)


def test_laplacian_matches_sum_of_second_derivatives(grid_2d):
    z = grid_2d
    got = d2_dxy2(z, D, D)
    # Interior cell (1,1): ((z[1,0]+z[1,2]-2z[1,1]) + (z[0,1]+z[2,1]-2z[1,1]))/D^2
    expected = ((z[1, 0] + z[1, 2] - 2 * z[1, 1]) + (z[0, 1] + z[2, 1] - 2 * z[1, 1])) / D / D
    assert np.allclose(got[1, 1], expected, atol=1e-5)


def test_laplacian_constant_field_is_zero():
    z = jnp.full((6, 6), 3.25)
    got = d2_dxy2(z, D, D)[2:-2, 2:-2]
    assert np.allclose(got, 0.0, atol=1e-6)


def test_laplacian_linear_field_is_zero():
    z = jnp.arange(8 * 9, dtype=jnp.float32).reshape(8, 9) * 0.7
    got = d2_dxy2(z, D, D)[2:-2, 2:-2]
    assert np.allclose(got, 0.0, atol=1e-4)


def test_d_uh_dx_interior(grid_2d):
    z = grid_2d
    h = jnp.ones_like(z) * 0.3
    got = d_uh_dx(z, h, D)
    expected = (z[1, 3] * h[1, 3] - z[1, 1] * h[1, 1]) / (2 * D)
    assert np.allclose(got[1, 2], expected, atol=1e-6)


def test_d_vh_dy_interior(grid_2d):
    z = grid_2d
    h = jnp.ones_like(z) * 0.3
    got = d_vh_dy(z, h, D)
    expected = (z[2, 2] * h[2, 2] - z[0, 2] * h[0, 2]) / (2 * D)
    assert np.allclose(got[1, 2], expected, atol=1e-6)