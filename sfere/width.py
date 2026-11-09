# Channel-width estimators for the dX-invariance diagnostic
# (scripts/dx_diagnostic.py). Question answered: is the emergent channel
# width, measured IN METRES, invariant under grid coarsening (physics-set)
# or does it scale with dX (grid-controlled)?
#
# A "channel" is masked as a cell whose channel field exceeds a field
# quantile. Primary field: unit-width discharge |u|*h; cross-check: bed
# shear t (see shear()). Width is then measured two independent ways, plus
# one threshold-free scale:
#   runs_width       - run-lengths of mask cells along X, per row (channels
#                      drain in Y, so X is the cross-channel direction)
#   edt_width        - 2x the mean medial radius from the distance transform
#                      (underestimates strongly meandering channels)
#   spectrum_length  - dominant cross-channel wavelength from the row-averaged
#                      power spectrum; a spacing, not a width
# Everything is measured on a borderless interior (the model's edge rows and
# columns carry boundary conditions, see sfere.operators), connected specks
# below AREA_MIN cells are dropped, and all lengths are returned in metres
# so results are directly comparable across dX.

from __future__ import annotations

import numpy as np
from scipy import ndimage

BORDER = 2      # cells stripped from every edge before measuring
AREA_MIN = 9    # kept mask components span >= 3x3 cells; drops 1-cell noise
QUANTILES = (0.90, 0.95, 0.99)


def interior(field: np.ndarray, border: int = BORDER) -> np.ndarray:
    """The field with `border` cells stripped from every edge, as float64."""
    if border == 0:
        return np.asarray(field, dtype=np.float64)
    return np.asarray(field, dtype=np.float64)[border:-border, border:-border]


def channel_mask(field: np.ndarray, q: float, border: int = BORDER) -> np.ndarray:
    """Boolean mask of cells above the `q`-quantile of the borderless field."""
    z = interior(field, border)
    return z >= np.quantile(z, q)


def _clean(mask: np.ndarray, area_min: int = AREA_MIN) -> np.ndarray:
    """Drop connected components smaller than `area_min` cells."""
    if area_min <= 1:
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:][sizes[1:] >= area_min] = True
    return keep[labels]


def runs_width(mask: np.ndarray, dX: float, area_min: int = AREA_MIN) -> np.ndarray:
    """Widths (m) of channel cells, one per along-X run per row.

    Channels drain in Y, so X is the cross-channel direction: every run of
    consecutive channel cells in a row is one cross-section of one channel.
    Returns a flat array of run lengths in metres; aggregate with
    np.median / np.mean / np.quantile. Empty array when the mask is empty.
    """
    m = _clean(mask, area_min)
    if not m.any():
        return np.empty(0)
    runs: list[np.ndarray] = []
    for row in m:
        pad = np.zeros(1, dtype=np.int8)
        padded = np.concatenate((pad, row.astype(np.int8), pad))
        d = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        runs.append(ends - starts)
    return np.concatenate(runs) * dX


def edt_width(mask: np.ndarray, dX: float, area_min: int = AREA_MIN) -> float:
    """Mean channel width (m): 2x the maximal medial radius, per component.

    For each kept component the distance-transform maximum is the medial
    radius (exactly width/2 for straight stripes and round blobs alike, and
    close to it for meandering channels), so width_i = 2*max; components are
    combined area-weighted. Returns nan for an empty (cleaned) mask.
    """
    m = _clean(mask, area_min)
    if not m.any():
        return float("nan")
    labels, n = ndimage.label(m)
    if n == 0:
        return float("nan")
    # False ring: edge-touching components then have their boundary counted,
    # which stops the transform from measuring across the whole array edge.
    dist = ndimage.distance_transform_edt(np.pad(m, 1))
    dist = dist[1:-1, 1:-1]
    widths = np.empty(n)
    weights = np.empty(n)
    for lbl in range(1, n + 1):
        inside = dist[labels == lbl]
        widths[lbl - 1] = 2.0 * inside.max()
        weights[lbl - 1] = inside.size
    return float(np.average(widths, weights=weights)) * dX


def spectrum_length(field: np.ndarray, dX: float, border: int = BORDER,
                    min_period_cells: int = 8) -> float:
    """Dominant cross-channel wavelength (m) from the row-averaged spectrum.

    Threshold-free companion to the mask estimators: the spectral peak in
    the cross-channel (X) direction is the channel spacing (wavelength =
    channel + gap). Periods below `min_period_cells` are skipped so the
    peak reflects structure, not 2-cell noise. Returns nan for a flat field.
    """
    z = interior(field, border)
    if not np.any(z - z.mean(axis=1, keepdims=True)):
        return float("nan")
    power = np.abs(np.fft.rfft(z - z.mean(axis=1, keepdims=True), axis=1)) ** 2
    power = power.mean(axis=0)
    n_cells = z.shape[1]
    k_max = n_cells // min_period_cells
    k_peak = 1 + int(np.argmax(power[1:k_max + 1]))
    return float(n_cells / k_peak) * dX


def measure_field(field: np.ndarray, dX: float, quantiles: tuple = QUANTILES,
                  area_min: int = AREA_MIN) -> dict:
    """All estimators for one channel field; quantile-keyed, metres."""
    out = {"spectrum": spectrum_length(field, dX)}
    for q in quantiles:
        mask = channel_mask(field, q)
        widths = runs_width(mask, dX, area_min)
        out[f"q{q:g}"] = {
            "runs_median": float(np.median(widths)) if widths.size else float("nan"),
            "runs_mean": float(widths.mean()) if widths.size else float("nan"),
            "runs_p10": float(np.quantile(widths, 0.10)) if widths.size else float("nan"),
            "runs_p90": float(np.quantile(widths, 0.90)) if widths.size else float("nan"),
            "n_runs": int(widths.size),
            "edt": edt_width(mask, dX, area_min),
        }
    return out


def shear(field_u: np.ndarray, field_v: np.ndarray, field_h: np.ndarray,
          field_d: np.ndarray, nb: float, nv: float, kk: float,
          rho: float, g: float) -> np.ndarray:
    """Bed shear t = rho*g/Ct^2*|u|^2 recomputed from saved arrays.

    Same formulas as sfere.model step 2/6 (Ct = nn^-1*h^(1/6),
    nn = nb + (nv-nb)*d/kk), in float64; float32 roundoff is irrelevant at
    quantile thresholds. Note this matches the model's diagnostics only up
    to their one-step staleness - irrelevant here.
    """
    u = np.asarray(field_u, dtype=np.float64)
    v = np.asarray(field_v, dtype=np.float64)
    h = np.maximum(np.asarray(field_h, dtype=np.float64), 0.0)
    nn = nb + (nv - nb) * np.asarray(field_d, dtype=np.float64) / kk
    uabs2 = u ** 2 + v ** 2
    return rho * g * (nn ** 2 / h ** (1.0 / 3.0)) * uabs2
