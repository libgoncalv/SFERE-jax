# Width estimators against synthetic fields with known channel geometry.

from __future__ import annotations

import numpy as np
import pytest

from sfere import width

Q = 0.9  # above the 40% channel fraction, so the quantile lands on the value


def stripe_field(n_rows: int, n_cols: int, period: int, width_cells: int) -> np.ndarray:
    """Parallel channels draining in Y: width_cells-of-period stripes in X."""
    x = np.arange(n_cols) % period
    x = np.where(x < width_cells, 1.0, 0.0).astype(float)
    return x[np.newaxis, :] * np.ones((n_rows, 1))


def test_runs_width_recovers_stripe_width_in_metres():
    # Same physical stripes sampled at dX=0.5 (400x1600) and dX=1.0 (200x800):
    # the estimators themselves must be scale-invariant in metres.
    for dX, n_rows, n_cols in ((0.5, 400, 1600), (1.0, 200, 800)):
        field = stripe_field(n_rows, n_cols, period=200, width_cells=80)
        widths = width.runs_width(width.channel_mask(field, Q), dX)
        assert widths.size > n_rows  # many cross-sections per row over rows
        assert np.median(widths) == pytest.approx(80 * dX, rel=0.02)


def test_edt_width_recovers_stripe_width_in_metres():
    for dX, n_rows, n_cols in ((0.5, 400, 1600), (1.0, 200, 800)):
        field = stripe_field(n_rows, n_cols, period=200, width_cells=80)
        w = width.edt_width(width.channel_mask(field, Q), dX)
        assert w == pytest.approx(80 * dX, rel=0.02)


def test_edt_width_handles_edge_touching_channels():
    # Regression: an edge-clipped stripe used to measure 2x its width
    # because the transform had no background cell beyond the array edge.
    field = stripe_field(400, 1600, period=200, width_cells=80)
    w = width.edt_width(width.channel_mask(field, Q), 0.5)
    assert w < 80 * 0.5 + 0.05  # area-weighted: mostly 40 m, clipped stripe 39 m


def test_spectrum_length_recovers_channel_spacing():
    for dX, n_rows, n_cols in ((0.5, 400, 1600), (1.0, 200, 800)):
        field = stripe_field(n_rows, n_cols, period=200, width_cells=80)
        spacing = width.spectrum_length(field, dX)
        assert spacing == pytest.approx(200 * dX, rel=0.05)


def test_measure_field_keys_and_scales():
    field = stripe_field(400, 1600, period=200, width_cells=80)
    out = width.measure_field(field, 0.5)
    assert out["spectrum"] == pytest.approx(100.0, rel=0.05)
    hi = out["q0.9"]
    assert hi["runs_median"] == pytest.approx(40.0, rel=0.02)
    assert hi["edt"] == pytest.approx(40.0, rel=0.05)
    assert hi["n_runs"] > 3000


def test_bimodal_threshold_gives_exact_mask():
    # Regression for the quantile-on-ties trap: on a two-valued field the
    # threshold must land between 0 and 1, not on 0 (mask = everything).
    field = stripe_field(400, 1600, period=200, width_cells=80)
    assert width.channel_mask(field, 0.5).all()  # q=0.5 sits inside the zeros, by value
    # interior 396x1596: one edge-clipped stripe of 78 + seven full of 80
    assert width.channel_mask(field, Q).sum() == 396 * (78 + 7 * 80)


def test_area_min_drops_isolated_specks():
    field = stripe_field(400, 1600, period=200, width_cells=80)
    field[10, 900] = 1.0  # isolated 1-cell speck
    widths = width.runs_width(width.channel_mask(field, Q), 0.5)
    assert not np.isclose(widths, 0.5).any()  # no 1-cell runs survive
    assert np.median(widths) == pytest.approx(40.0, rel=0.02)


def test_empty_mask_returns_empty_and_nan():
    field = np.zeros((200, 800))
    mask = np.zeros_like(width.channel_mask(field, Q), dtype=bool)
    assert width.runs_width(mask, 0.5).size == 0
    assert np.isnan(width.edt_width(mask, 0.5))
    assert np.isnan(width.spectrum_length(field, 0.5))


def test_shear_matches_model_formula():
    h = np.full((10, 10), 0.02)
    d = np.zeros((10, 10))
    u = np.full((10, 10), 0.3)
    v = np.zeros((10, 10))
    t = width.shear(u, v, h, d, nb=0.016, nv=0.2, kk=1500.0, rho=1000.0, g=9.81)
    expected = 1000.0 * 9.81 * 0.016 * 0.016 / h[0, 0] ** (1 / 3) * 0.3 ** 2
    assert np.allclose(t, expected)
