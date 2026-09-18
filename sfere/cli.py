# CLI entry point: python -m sfere run [...]

from __future__ import annotations

import argparse

from .config import GradientOptions, GridConfig, PhysParams, RunConfig, SimConfig
from .io import save_all
from .model import Params
from .simulate import run_frames, stack
from .state import init_conditions
from .viz import save_animation, save_figure, save_overview


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sfere", description="SFERE model (JAX port)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation")
    run.add_argument("--endtime", type=float, default=70580.0, help="simulated tides")
    run.add_argument("--frames", type=int, default=100, help="number of saved frames")
    run.add_argument("--dt", type=float, default=0.0125, help="time step [s]")
    run.add_argument("--grid", default="2048x1024", help="Grid_Width x Grid_Height (powers of 2)")
    run.add_argument("--dx", type=float, default=0.5, help="cell width [m]")
    run.add_argument("--dy", type=float, default=0.5, help="cell height [m]")
    run.add_argument("--run", type=int, default=9, help="RUN number (output naming)")
    run.add_argument("--outdir", default="RUN9", help="output directory")
    run.add_argument("--seed", type=int, default=2, help="initial-condition RNG seed")
    run.add_argument("--no-vegetation", action="store_true", help="disable vegetation dynamics")
    for flag in ("D0", "Hin", "DifD", "pD"):
        run.add_argument(f"--gradient-{flag.lower()}", action="store_true",
                         help=f"enable the across-width gradient in {flag}")
    run.add_argument("--no-figure", action="store_true")
    run.add_argument("--no-animation", action="store_true")
    run.add_argument("--no-overview", action="store_true")
    run.add_argument("--no-save", action="store_true")
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
    bx, bn = 16, W // 16
    by, bn_y = 16, H // 16
    grid = GridConfig(Block_Size_X=bx, Block_Size_Y=by, Block_Number_X=bn, Block_Number_Y=bn_y,
                      dX=args.dx, dY=args.dy)

    phys = PhysParams()
    gradients = GradientOptions(D0=args.gradient_d0, Hin=args.gradient_hin,
                                DifD=args.gradient_difd, pD=args.gradient_pd)
    sim = SimConfig(EndTime=args.endtime, NumFrames=args.frames, dT=args.dt)
    run = RunConfig(RUN=args.run, SUBRUNa=1, SUBRUNb=1, outdir=args.outdir, seed=args.seed,
                    vegetation=not args.no_vegetation)

    params = Params(phys=phys, gradients=gradients, dX=grid.dX, dY=grid.dY, dT=sim.dT)

    print(f"Grid dimensions: {grid.Grid_Width} x {grid.Grid_Height}")
    state = init_conditions(phys, grid, vegetation=run.vegetation, seed=run.seed)

    states, _ = run_frames(state, params, sim)
    stacks = stack(states)

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
    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()