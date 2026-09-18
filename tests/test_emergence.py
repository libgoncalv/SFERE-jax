# Slow smoke test (marked, run manually): channels emerge from random seedlings
# on a reduced grid after a short simulation.

import jax
import numpy as np
import pytest

from sfere.config import GradientOptions, GridConfig, PhysParams, SimConfig
from sfere.model import Params
from sfere.simulate import run_frames
from sfere.state import init_conditions

pytestmark = pytest.mark.slow


def test_emergence_smoke():
    grid = GridConfig(Block_Size_X=16, Block_Size_Y=16, Block_Number_X=8, Block_Number_Y=4)
    phys = PhysParams()
    params = Params(phys=phys, gradients=GradientOptions(), dX=grid.dX, dY=grid.dY, dT=0.0125)
    sim = SimConfig(EndTime=5000.0, NumFrames=2, dT=0.0125)

    state = init_conditions(phys, grid, vegetation=True, seed=2)
    states, _ = run_frames(state, params, sim, verbose=False)
    final = states[-1]

    for name in ("u", "v", "h", "s", "d"):
        assert np.isfinite(np.asarray(getattr(final, name))).all(), name

    # Sediment must have been redistributed: no longer spatially uniform
    s_final = np.asarray(final.s)
    interior = s_final[2:-2, 2:-2]
    assert interior.std() > 0.0
    # Vegetation grew from seedlings (net growth before mortality)
    assert np.asarray(final.d).max() > 0.0