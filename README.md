# SFERE
SFERE (Scale-dependent Feedback Recursion). Mathematical model to explain self-organization of complex spatial patterns from recursion of a scale-dependent feedback at multiple nested scales. First applied to self-organization of branching channel networks in coastal wetlands (van de Vijsel et al. 2023, Nature Communications).

## JAX implementation

This branch (`jax-rewrite`) contains a pure-Python/JAX port of the original OpenCL model. All numerics are expressed as array operations and compiled with `jax.jit`/`lax.fori_loop`; no OpenCL C, no string-built kernels.

### Setup

```bash
conda create -n sfere-jax -c conda-forge --override-channels python=3.12 -y
conda run -n sfere-jax pip install -r requirements.txt   # jax[cuda13], numpy, matplotlib, pytest
ffmpeg install via apt is assumed (for MP4 output)
```

### Running

```bash
conda run -n sfere-jax python -m sfere run                       # default RUN9-equivalent (2048x1024, 100 frames)
conda run -n sfere-jax python -m sfere run --no-vegetation       # vegetation-off control (Suppl. Fig. 2)
conda run -n sfere-jax python -m sfere run --grid 512x256 --endtime 5000   # quick test run
conda run -n sfere-jax python -m sfere sweep --para "nv:0.016,0.2" --devices 2   # sweep on 2 GPUs
```

Notebook use: open `notebooks/SFERE.ipynb` (Jupyter or VS Code). It configures a
notebook-scale run, shows a live 8-panel figure updated in place after every
frame, builds an inline animation of all frames, and saves with the CLI's
filenames:

```python
from sfere.simulate import Simulation
from sfere.live import LiveFigure, html_animation

simulation = Simulation(state, params, sim)   # after init_conditions()
live = LiveFigure(simulation, run)
for _ in range(sim.NumFrames):
    simulation.advance()
    live.show()
html_animation(simulation, run)
simulation.save(run)
```

Outputs land in `RUN9/` with the same filenames as the original notebook (PNG, MP4, `_overview.png`, `.npy` saves). Tests: `conda run -n sfere-jax python -m pytest tests/` (add `-m slow` for the emergence smoke test).

### Layout

| Path | Role |
|---|---|
| `sfere/config.py` | Dataclasses: `PhysParams` (RUN9 defaults, MORF-premultiplied), `GridConfig`, `SimConfig`, `RunConfig`, `GradientOptions` |
| `sfere/state.py` | `State` pytree (the 8 variables as 2D arrays) + initial conditions (numpy RNG seed 2, matching the original) |
| `sfere/operators.py` | Port of `HydroFunctions_iPy.cl`: stencil operators + all 5 boundary conditions |
| `sfere/model.py` | `step()`: the 4 equation blocks (momentum → continuity → sediment → vegetation), exact ordering of the original kernel |
| `sfere/simulate.py` | jit-compiled frame loop (`lax.fori_loop`), host-side frame capture |
| `sfere/io.py` / `sfere/viz.py` | `.npy` saves + 8-panel figure / MP4 animation / mean-min-max overview |
| `sfere/cli.py` / `sfere/sweep.py` | `run` and `sweep` subcommands; gradient switches are runtime flags |
| `sfere/live.py` | Notebook display helpers: in-place live 8-panel figure (`LiveFigure`) + inline animation (`html_animation`); IPython imported lazily |
| `notebooks/SFERE.ipynb` | Interactive notebook: run, watch the evolution live, animate, save (replaces the original's single-cell experience) |

### Notes for newcomers

- **Coordinate flip** (inherited from the original): in the code, X runs left→right and Y top→bottom; in the manuscript these axes (and `u`/`v`) are swapped. Consistent throughout; output unaffected. Do not "fix" it.
- **Ordering**: `h` is clamped to `≥ HCrit` *before* the momentum equations; diagnostics `c`, `t` are computed from *step-start* flow (stale), matching the original kernel.
- **Boundary assignment** (per variable, as in the kernel): `u`,`v` — reflecting walls (top/left/right) + persistent-flux outflow (bottom); `h`,`d` — Neumann; `s` — Neumann, Dirichlet-0 at the bottom.
- The original's compile-time `Gradient_*` switches are runtime flags (`--gradient-d0` etc.); gradient values follow the same across-width log-interpolation.
- Unlike the original OpenCL kernel (racy mid-kernel reads), JAX evaluates derivatives sequentially from consistent fields — same equations, cleaner semantics.

## Files
#### clPy.ComplexChannelPatterns_Fig1f_RUN9.ipynb
This is the main model script. This is the default version of the SFERE model, i.e. the version used to create Fig. 1f in the manuscript (van de Vijsel et al. 2023, Nature Communications). The script can be opened and run in Jupyter Notebook.

#### HydroFunctions_iPy.cl
This script contains functions (e.g., derivatives and boundary conditions) which are called upon in the main model script. Make sure to place this script in the same directory as the main model script.

#### RUN9
A folder named "RUN9" should be created and placed in the same directory as clPy.ComplexChannelPatterns_Fig1f_RUN9.ipynb and HydroFunctions_iPy.cl. The model results are stored in this folder.

![SupplFig2_dpi=250](https://github.com/RCvandeVijsel/SFERE/assets/130892539/54ec43a5-d2a1-48a6-ac18-43fd80ae0f91)
Supplementary Fig. 2 (van de Vijsel et al. 2023) shows channel network development in the default configuration of SFERE (with vegetation, right column), and without vegetation dynamics (left column). Vegetation clearly increases the extent and complexity of channel networks.

## More information
More information about the functioning of the SFERE model can be found in:
- The manuscript (van de Vijsel et al. 2023, Nature Communications)
- Commented lines in the model scripts (clPy.ComplexChannelPatterns_Fig1f_RUN9.ipynb and HydroFunctions_iPy.cl) themselves
- The full dataset and its readme file, underlying the manuscript (van de Vijsel et al. 2023, Nature Communications). This dataset can be found via https://doi.org/10.4121/8d361887-ec02-4472-a8eb-a9d0f3eacfd6.

## References
van de Vijsel, R.C., van Belzen, J., Bouma, T.J., van der Wal, D., Borsje, B.W., Temmerman, S., Cornacchia, L., Gourgue, O., van de Koppel, J. Vegetation controls on channel network complexity in coastal wetlands. Nature Commununications 14, 7158 (2023). https://doi.org/10.1038/s41467-023-42731-3.
