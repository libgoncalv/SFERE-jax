# Tests for the 5 boundary conditions, replicating HydroFunctions_iPy.cl's
# formulas and branch precedence (top row first; corners follow the top-row
# or bottom-row branch; side columns apply to interior rows only).

import jax.numpy as jnp
import numpy as np

from sfere.operators import (
    DirichletBoundaries,
    NeumanBoundaries,
    PeriodicBoundaries,
    PersistentFluxBoundaries,
    PersistentFluxBoundaries_bottom,
    ReflectingBoundaries,
)


def make_grid(h=6, w=7):
    return jnp.arange(h * w, dtype=jnp.float32).reshape(h, w) / 3.0


def test_periodic():
    z = make_grid()
    got = PeriodicBoundaries(z)
    assert np.allclose(got[0, :], z[4, :])        # top <- row H-2
    assert np.allclose(got[5, :], z[1, :])        # bottom <- row 1
    assert np.allclose(got[1:5, 0], z[1:5, 5])    # left interior rows <- col W-2
    assert np.allclose(got[1:5, 6], z[1:5, 1])    # right interior rows <- col 1


def test_neumann():
    z = make_grid()
    got = NeumanBoundaries(z)
    assert np.allclose(got[0, :], z[1, :])
    assert np.allclose(got[5, :], z[4, :])
    assert np.allclose(got[1:5, 0], z[1:5, 1])
    assert np.allclose(got[1:5, 6], z[1:5, 5])
    # Interior untouched
    assert np.allclose(got[2, 2], z[2, 2])


def test_reflecting():
    u = make_grid()
    v = make_grid() + 10.0
    gu, gv = ReflectingBoundaries(u, v)
    # Top row: u copied, v negated
    assert np.allclose(gu[0, :], u[1, :])
    assert np.allclose(gv[0, :], -v[1, :])
    # Side columns: u negated, v copied
    assert np.allclose(gu[1:5, 0], -u[1:5, 1])
    assert np.allclose(gv[1:5, 0], v[1:5, 1])
    assert np.allclose(gu[1:5, 6], -u[1:5, 5])
    assert np.allclose(gv[1:5, 6], v[1:5, 5])


def test_persistent_flux():
    z = make_grid()
    got = PersistentFluxBoundaries(z)
    assert np.allclose(got[0, :], 2 * z[1, :] - z[2, :])
    assert np.allclose(got[5, :], 2 * z[4, :] - z[3, :])
    assert np.allclose(got[1:5, 0], 2 * z[1:5, 1] - z[1:5, 2])
    assert np.allclose(got[1:5, 6], 2 * z[1:5, 5] - z[1:5, 4])


def test_persistent_flux_bottom_only():
    z = make_grid()
    got = PersistentFluxBoundaries_bottom(z)
    assert np.allclose(got[5, :], 2 * z[4, :] - z[3, :])
    assert np.allclose(got[0, :], z[0, :])  # other rows untouched


def test_dirichlet():
    z = make_grid()
    got = DirichletBoundaries(z, 0.25)
    assert np.allclose(got[0, :], 0.25)
    assert np.allclose(got[5, :], 0.25)
    assert np.allclose(got[:, 0], 0.25)
    assert np.allclose(got[:, 6], 0.25)
    assert np.allclose(got[2, 2], z[2, 2])