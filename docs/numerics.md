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

## Physics equivalence

The improved path is expected to change *how the states are stepped*, not
*what the model converges to*: at the reference settings the improved run
reproduces the channel width measured by the dX-invariance diagnostic
(~12 m on the 1024x512 m domain; see `DIAGNOSTIC/verdict.md`). If a settings
combination changed the widths materially, that difference would itself be
a finding - treat it as such, not as noise.

## Usage

```bash
conda run -n sfere-jax sfere run --improve-friction --improve-advection \
    --dt 0.05 --dx 1.0 --grid 1024x512 --outdir IMPROVED/dx1
```

Each flag works alone; output names carry `_IMPf` / `_IMPa` / `_IMPfa`
according to the switches used.
