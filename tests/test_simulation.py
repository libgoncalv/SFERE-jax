# Simulation driver tests: incremental advance == run_frames, resumability,
# saving with the original filenames, and the reusable figure machinery
# (draw_figure/set_frame) that the notebook display layer builds on.

import numpy as np
import pytest

from sfere import viz
from sfere.config import GradientOptions, GridConfig, PhysParams, RunConfig, SimConfig
from sfere.model import Params
from sfere.simulate import Simulation, run_frames, stack
from sfere.state import init_conditions

GRID = GridConfig(Block_Size_X=16, Block_Size_Y=16, Block_Number_X=2, Block_Number_Y=2)
PHY = PhysParams()
PARAMS = Params(phys=PHY, gradients=GradientOptions(), dX=GRID.dX, dY=GRID.dY, dT=0.0125)


def make_sim(num_frames: int = 2, endtime: float = 1.0) -> Simulation:
    sim = SimConfig(EndTime=endtime, NumFrames=num_frames, dT=0.0125)
    state = init_conditions(PHY, GRID, vegetation=True, seed=2)
    return Simulation(state, PARAMS, sim)


def assert_frames_equal(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        for name in fa:
            assert np.array_equal(fa[name], fb[name]), name


def test_latest_frame_is_initial_state_before_advance():
    sim = make_sim()
    latest = sim.latest_frame()
    state = init_conditions(PHY, GRID, vegetation=True, seed=2)
    for name in ("u", "v", "h", "b", "s", "d", "c", "t"):
        assert np.array_equal(latest[name], np.asarray(getattr(state, name)))


def test_advance_matches_run_frames():
    sim = SimConfig(EndTime=1.0, NumFrames=2, dT=0.0125)
    expected, _ = run_frames(init_conditions(PHY, GRID, vegetation=True, seed=2),
                             PARAMS, sim, verbose=False)
    got = make_sim()
    assert_frames_equal([got.advance()], expected[:1])
    assert_frames_equal([got.advance()], expected[1:])
    assert got.n_frames == 2


def test_advance_is_resumable():
    """advance() in separate cells must equal one continuous run (no re-jit drift)."""
    a = make_sim()
    a.advance()
    a.advance()

    b = make_sim()
    b.advance(2)

    sim = SimConfig(EndTime=1.0, NumFrames=2, dT=0.0125)
    c, _ = run_frames(init_conditions(PHY, GRID, vegetation=True, seed=2), PARAMS, sim, verbose=False)
    assert_frames_equal(a.frames, c)
    assert_frames_equal(b.frames, c)


def test_time_bookkeeping():
    sim = make_sim()
    assert sim.tides_per_frame == pytest.approx(sim.sim.steps_per_frame * sim.sim.dT)
    assert sim.tides_elapsed == 0
    sim.advance()
    assert sim.n_frames == 1
    assert sim.tides_elapsed == pytest.approx(sim.tides_per_frame)


def test_save_no_frames_raises():
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.save(RunConfig(outdir="RUN9"))


def test_save_writes_original_filenames(tmp_path):
    sim = make_sim()
    sim.advance(2)
    run = RunConfig(RUN=9, outdir=str(tmp_path))
    paths = sim.save(run, animation=False)  # ffmpeg renders in save_animation; covered by viz users

    for name in ("u", "v", "h", "b", "s", "d"):
        assert (tmp_path / f"{run.ModelName}_SUBRUNa1_SUBRUNb1_{name}.npy").exists()
        assert (tmp_path / f"{run.ModelName}_SUBRUNa1_SUBRUNb1_{name}s.npy").exists()
    assert (tmp_path / f"{run.ModelName}_SUBRUNa1_SUBRUNb1.png").exists()
    assert (tmp_path / f"{run.ModelName}_SUBRUNa1_SUBRUNb1_overview.png").exists()
    assert paths["figure"].endswith(".png")
    assert set(paths) == {"figure", "overview"}


def test_draw_and_set_frame_update_panels():
    """The reusable figure machinery: set_frame mutates panels and title in place."""
    plt = pytest.importorskip("matplotlib.pyplot")
    sim = make_sim()
    sim.advance(2)
    stacks = stack(sim.frames)

    fig, handles = viz.draw_figure(viz._panels(stacks, 0), "base")
    viz.set_frame(handles, viz._panels(stacks, 1), "moved")
    assert np.array_equal(handles["panels"][0].get_array(), stacks["s"][1])
    assert np.array_equal(handles["panels"][3].get_array(), viz.panel_values(**sim.frames[1])["b"])
    assert handles["title"].get_text() == "moved"
    plt.close(fig)


def test_titles_match_original_format():
    run = RunConfig(RUN=9)
    assert viz.run_title(run) == f"{run.ModelName}_SUBRUNa1_SUBRUNb1"
    title = viz.time_title(viz.run_title(run), elapsed_tides=1000.0, end_tides=2000.0, facTY=1.0)
    assert title == f"{run.ModelName}_SUBRUNa1_SUBRUNb1       -       Time: 1000 of 2000 yr"


def test_live_figure_updates_display_id():
    """LiveFigure pushes one display and updates it; requires IPython."""
    pytest.importorskip("IPython")

    from sfere.live import LiveFigure

    sim = make_sim()
    live = LiveFigure(sim, RunConfig(RUN=9))
    live.show()
    assert live._handle is not None
    sim.advance()
    live.show()
    assert live._handles["title"].get_text().endswith("yr")
