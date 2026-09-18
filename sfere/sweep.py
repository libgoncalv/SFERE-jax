# Parameter sweeps: run multiple (SUBRUNa, SUBRUNb) configurations, optionally
# distributed across GPUs. Each sweep point sets one physical parameter
# (e.g. PARa -> nv, PARb -> Qs) to a value from a list, mirroring the
# original's commented-out PARa_arr/PARb_arr loops.

from __future__ import annotations

import argparse
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import GradientOptions, GridConfig, PhysParams, RunConfig, SimConfig
from .io import save_all
from .model import Params
from .simulate import run_frames, stack
from .state import init_conditions
from .viz import save_figure, save_overview

# Sweepable parameter names (fields of PhysParams).
SWEEPABLE = tuple(f.name for f in dataclasses.fields(PhysParams) if f.name != "MORF")


def _parse_grid(spec: str) -> GridConfig:
    w, h = (int(v) for v in spec.lower().split("x"))
    for v in (w, h):
        if v & (v - 1) or v < 16:
            raise SystemExit(f"grid dimension {v} must be a power of 2, >= 16")
    return GridConfig(Block_Size_X=16, Block_Size_Y=16,
                      Block_Number_X=w // 16, Block_Number_Y=h // 16)


def run_point(job: dict) -> str:
    """Run one sweep point in this process (must be a process worker target)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", job["mem_fraction"])
    # Lazy import so each spawned process binds its own CUDA context.
    import jax

    if job["device"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = job["device"]

    phys = PhysParams(**job["phys_overrides"])
    grid = _parse_grid(job["grid"])
    gradients = GradientOptions(**job["gradients"])
    sim = SimConfig(EndTime=job["endtime"], NumFrames=job["frames"], dT=job["dt"])
    run = RunConfig(RUN=job["run"], SUBRUNa=job["subruna"], SUBRUNb=job["subrunb"],
                    outdir=job["outdir"], seed=job["seed"], vegetation=job["vegetation"])
    params = Params(phys=phys, gradients=gradients, dX=grid.dX, dY=grid.dY, dT=sim.dT)

    state = init_conditions(phys, grid, vegetation=run.vegetation, seed=run.seed)
    frames, _ = run_frames(state, params, sim, verbose=False)
    stacks = stack(frames)
    save_all(stacks, run)
    save_figure(stacks, run)
    save_overview(stacks, run, phys.facTY, frame_tides=sim.EndTime / sim.NumFrames)
    return f"{job['label']}: done on device {jax.devices()[0]}"


def cmd_sweep(args: argparse.Namespace) -> None:
    grid = _parse_grid(args.grid)

    pa_name, pa_vals = _parse_axis(args.para)
    pb_name, pb_vals = _parse_axis(args.parb)
    if pa_name not in SWEEPABLE or pb_name not in SWEEPABLE:
        raise SystemExit(f"sweepable parameters: {', '.join(SWEEPABLE)}")

    n_dev = args.devices
    jobs = []
    for ia, va in enumerate(pa_vals, start=1):
        for ib, vb in enumerate(pb_vals, start=1):
            jobs.append(dict(
                phys_overrides={pa_name: float(va), pb_name: float(vb)},
                gradients={"D0": args.gradient_d0, "Hin": args.gradient_hin,
                           "DifD": args.gradient_difd, "pD": args.gradient_pd},
                endtime=args.endtime, frames=args.frames, dt=args.dt,
                grid=args.grid, run=args.run, subruna=ia, subrunb=ib,
                outdir=args.outdir, seed=args.seed, vegetation=not args.no_vegetation,
                device=(str(len(jobs) % n_dev) if n_dev > 0 else None),
                mem_fraction=f"{args.mem_fraction:.2f}",
                label=f"SUBRUNa{ia}_SUBRUNb{ib} ({pa_name}={va}, {pb_name}={vb})",
            ))

    print(f"sweep: {len(jobs)} points on {n_dev if n_dev > 0 else 'shared'} device(s)")
    if n_dev <= 0:
        for job in jobs:
            print(run_point(job))
    else:
        with ProcessPoolExecutor(max_workers=n_dev) as pool:
            futures = [pool.submit(run_point, job) for job in jobs]
            for fut in as_completed(futures):
                print(fut.result(), flush=True)


def _parse_axis(spec: str) -> tuple[str, list[float]]:
    """Parse 'name:val1,val2,...' into (name, values). Empty name -> dummy axis."""
    if spec.strip() in ("", "-"):
        return "Qs", [PhysParams().Qs]
    name, vals = spec.split(":", 1)
    return name.strip(), [float(v) for v in vals.split(",")]


def register_sweep(sub) -> None:
    sp = sub.add_parser("sweep", help="parameter sweep (SUBRUNa x SUBRUNb)")
    _add_run_common_into(sp)
    sp.add_argument("--para", default="-", help="axis a: 'name:v1,v2,...' (e.g. 'nv:0.016,0.05,0.2')")
    sp.add_argument("--parb", default="-", help="axis b: 'name:v1,v2,...'")
    sp.add_argument("--devices", type=int, default=0, help="number of GPUs (0 = sequential on one)")
    sp.add_argument("--mem-fraction", type=float, default=0.80, help="XLA memory fraction per device")
    sp.set_defaults(func=cmd_sweep)


def _add_run_common_into(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--endtime", type=float, default=70580.0)
    sp.add_argument("--frames", type=int, default=100)
    sp.add_argument("--dt", type=float, default=0.0125)
    sp.add_argument("--grid", default="2048x1024")
    sp.add_argument("--run", type=int, default=9)
    sp.add_argument("--outdir", default="RUN9")
    sp.add_argument("--seed", type=int, default=2)
    sp.add_argument("--no-vegetation", action="store_true")
    for flag in ("D0", "Hin", "DifD", "pD"):
        sp.add_argument(f"--gradient-{flag.lower()}", action="store_true")