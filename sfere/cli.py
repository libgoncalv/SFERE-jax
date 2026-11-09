# CLI entry point: python -m sfere run|sweep [...]

from __future__ import annotations

import argparse

from .config import GradientOptions, GridConfig, NumericsOptions, PhysParams, RunConfig, SimConfig
from .io import save_all
from .model import Params
from .simulate import run_frames, stack
from .state import init_conditions
from .viz import save_animation, save_figure, save_overview


def _add_run_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--endtime", type=float, default=70580.0, help="simulated tides")
    sp.add_argument("--frames", type=int, default=100, help="number of saved frames")
    sp.add_argument("--dt", type=float, default=0.0125, help="time step [s]")
    sp.add_argument("--grid", default="2048x1024", help="Grid_Width x Grid_Height (powers of 2)")
    sp.add_argument("--run", type=int, default=9, help="RUN number (output naming)")
    sp.add_argument("--outdir", default="RUN9", help="output directory")
    sp.add_argument("--seed", type=int, default=2, help="initial-condition RNG seed")
    sp.add_argument("--no-vegetation", action="store_true", help="disable vegetation dynamics")
    for flag in ("D0", "Hin", "DifD", "pD"):
        sp.add_argument(f"--gradient-{flag.lower()}", action="store_true",
                        help=f"enable the across-width gradient in {flag}")
    sp.add_argument("--improve-friction", action="store_true",
                    help="semi-implicit friction (departure from the original kernel; "
                         "removes its dT cap - see docs/numerics.md)")
    sp.add_argument("--improve-advection", action="store_true",
                    help="upwind momentum advection stencil (departure from the "
                         "original kernel - see docs/numerics.md)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sfere", description="SFERE model (JAX port)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation")
    _add_run_common(run)
    run.add_argument("--dx", type=float, default=0.5, help="cell width [m]")
    run.add_argument("--dy", type=float, default=0.5, help="cell height [m]")
    run.add_argument("--no-figure", action="store_true")
    run.add_argument("--no-animation", action="store_true")
    run.add_argument("--no-overview", action="store_true")
    run.add_argument("--no-save", action="store_true")
    run.set_defaults(func=cmd_run)

    from .sweep import register_sweep

    register_sweep(sub)
    return p


def _parse_grid(spec: str) -> tuple[int, int]:
    w, h = (int(v) for v in spec.lower().split("x"))
    for v in (w, h):
        if v & (v - 1) or v < 16:
            raise SystemExit(f"grid dimension {v} must be a power of 2, >= 16")
    return w, h


def cmd_run(args: argparse.Namespace) -> None:
    W, H = _parse_grid(args.grid)

    # Factor grid config so that Block_Size=16 and powers of 2 give W x H
    grid = GridConfig(Block_Size_X=16, Block_Size_Y=16,
                      Block_Number_X=W // 16, Block_Number_Y=H // 16,
                      dX=args.dx, dY=args.dy)

    phys = PhysParams()
    gradients = GradientOptions(D0=args.gradient_d0, Hin=args.gradient_hin,
                                DifD=args.gradient_difd, pD=args.gradient_pd)
    numerics = NumericsOptions(semi_implicit_friction=args.improve_friction,
                               upwind_advection=args.improve_advection)
    sim = SimConfig(EndTime=args.endtime, NumFrames=args.frames, dT=args.dt)
    tag = ""
    if args.improve_friction or args.improve_advection:
        parts = [name for name, on in (("semi-implicit friction", args.improve_friction),
                                       ("upwind advection", args.improve_advection)) if on]
        tag = "IMP" + ("f" if args.improve_friction else "") + ("a" if args.improve_advection else "")
        print(f"numerics: improved layer ({', '.join(parts)}) - deviates from the "
              "original kernel, see docs/numerics.md")
    run = RunConfig(RUN=args.run, SUBRUNa=1, SUBRUNb=1, outdir=args.outdir, seed=args.seed,
                    vegetation=not args.no_vegetation, tag=tag)

    params = Params(phys=phys, gradients=gradients, dX=grid.dX, dY=grid.dY, dT=sim.dT)

    print(f"Grid dimensions: {grid.Grid_Width} x {grid.Grid_Height}")
    state = init_conditions(phys, grid, vegetation=run.vegetation, seed=run.seed)

    frames, _ = run_frames(state, params, sim, numerics=numerics if tag else None)
    stacks = stack(frames)

    if not args.no_save:
        save_all(stacks, run)
    if not args.no_figure:
        print(save_figure(stacks, run))
    if not args.no_animation:
        print(save_animation(stacks, run))
    if not args.no_overview:
        print(save_overview(stacks, run, phys.facTY, frame_tides=sim.EndTime / sim.NumFrames))
    print("done")


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()