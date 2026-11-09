# Numerics variants: the faithful port vs the improvement layer

## Why two layers

`sfere/model.py::step` is a line-by-line port of the original OpenCL
`SimulationKernel` (upstream [RCvandeVijsel/SFERE](https://github.com/RCvandeVijsel/SFERE));
its purpose is provenance - a runnable, bit-comparable definition of the
original physics. The improvement layer (`sfere/numerics.py`) therefore never
edits that file: every deviation lives beside it, is individually opt-in, and
every improved run is labeled in its output filenames (e.g. base name
`..._IMPfa`, via `RunConfig.tag`). With all switches off, `step_improved`
delegates to `model.step` itself, so the two layers cannot silently mix.

## Deviation 1: semi-implicit friction (`--improve-friction`)

Original (`sfere/model.py`, momentum block), explicit forward Euler:

```
du = (... - g_Ct2*|u|*u/h ...)        # friction rate lambda = g_Ct2*|u|/h
u  = u_prev + dT*du                   # per-step gain (1 - dT*lambda)
```

Improved (`sfere/numerics.py`), backward Euler on that linear term, with the
same step-start lagging of |u| the kernel already uses:

```
u = (u_prev + dT*(du - g_Ct2*|u|*u/h)) / (1 + dT*lambda)   # gain 1/(1+dT*lambda) <= 1
```

Effect: the stability bound `dT < 2/lambda` disappears. `lambda ~ 118 /s` in
vegetated thin-film cells (h = HCrit = 1 mm, Manning n = nv = 0.2), which
capped `dT` at ~0.017 s in the reference RUN9 setup (dT = 0.0125 runs; 0.025
diverges) - and that cap contains no dX, so grid coarsening could never
reduce the step count. After the fix, `dT` is set by the viscous bound
`dT <~ dX^2/(4*DifU)` (0.125 s at dX = 0.5 m), which grows with dX.

## Deviation 2: upwind momentum advection (`--improve-advection`)

Original (momentum advection), centred stencils (d_dx / d_dy: symmetric
averaging of both neighbours). Centred + forward Euler is unconditionally
unstable for pure advection; the kernel survives only because the eddy
diffusion `DifU` damps the barely-resolved scales. That balance is measured
by the grid Peclet number `Pe = |U|*dX/DifU`: ~0.3 at dX = 0.5 m, growing
linearly with dX. Past Pe ~ 1-3 the port tends to checkerboard oscillations
before diverging, which puts a hard wall on coarsening.

Improved: the advection stencil reads the upstream cell (flow-direction
stencil, `_upwind_x/_upwind_y`). It carries its own numerical diffusion
(~|u|*dX/2), removing that wall for practical dX; the price of extra
diffusion can later be repaid by lowering `DifU` if sharper fronts are
wanted. Momentum only: continuity, sediment and vegetation keep the kernel
stencils exactly.

## What is NOT changed

Everything else is the port, line for line: pressure gradient (dO_dx/dy),
the h >= HCrit clamp before momentum, all five boundary conditions with the
kernel's branch precedence, the continuity/sediment/vegetation blocks, the
stale-uabs/Ct diagnostics, `jax` evaluation ordering, and the initial
conditions (numpy seed 2).

## Guarantees (tests/test_numerics.py)

1. All switches off: `step_improved` returns bit-identical arrays to
   `model.step` on identical states.
2. At dT = 0.05 (past the old cap, on a stiff vegetated thin-film state):
   the explicit friction diverges within ~60 steps, the semi-implicit form
   stays finite for 300.
3. The switched paths stay finite and actually differ from the port.

## Physics equivalence and measured gains

Measured on this workstation (1x CUDA device), same 1024x512 m domain,
EndTime 70580, improved path with both switches on:

| run | dX | dT | steps | wall | runs-median width | finite |
|---|---|---|---|---|---|---|
| port reference | 0.5 m | 0.0125 | 5.646M | ~23 min | 12.00 m | yes |
| improved | 0.5 m | 0.0125 | 5.646M | ~23 min | **11.00 m** | yes |
| improved | 0.5 m | 0.05 | 1.412M | ~7 min | **11.50 m** | yes |
| improved | 1.0 m | 0.05 | 1.412M | ~3 min | **17.00 m** | yes |
| improved | 2.0 m | 0.1 / 0.025 | - | - | - | **no** (below) |

Physics equivalence: at dT=0.0125 the improved path gives 11.0 m vs the
port's 12.0 m - within the estimator's threshold sensitivity band - and the
sediment fields correlate 0.90 with means 0.2281 vs 0.2283 m. Same model
answers. The gain: dT can be raised 4x (0.0125 -> 0.05) with the SAME
physical EndTime: 3.4x less wall time at dX=0.5, ~7.7x at dX=1.0. The
morphological damping rates stay far inside their explicit bounds
(edge erosion rate lam_s*DT <~ 0.01 throughout, probed).

## Limit found at dX = 2: positive-depth continuity (next deviation)

At dX=2.0 the run goes non-finite before completion, at both dT=0.1
(negative min(h) = -0.0005 m from ~step 90k, u non-finite by ~128k) and
dT=0.025 (probe finite through 25% of the run, saved final state
non-finite). Mechanism: forward-Euler continuity with centred fluxes can
overshoot a converging drying cell below zero depth; the kernel's
step-start clamp (h >= HCrit) then "repairs" it with an implicit water
source, which recurs often enough at the unresolved scales to destabilize
the state. Not friction, not advection, and not fixed by smaller dT.

Remedy (future deviation #3, deliberately not in this layer to keep its
scope auditable): positivity-preserving continuity - flux limiting or a
sub-stepped hydro update - after which dX~2-3 m (the width/4~3 m coarsening
target from the diagnostic) becomes reachable. Validated window for THIS
layer: dX <= 1 m with dT <= 0.05.

## Usage

```bash
conda run -n sfere-jax sfere run --improve-friction --improve-advection \
    --dt 0.05 --dx 1.0 --grid 1024x512 --outdir IMPROVED/dx1
```

Each flag works alone; output names carry `_IMPf` / `_IMPa` / `_IMPfa`
according to the switches used.
