#!/usr/bin/env python
# dX-invariance diagnostic for the channel width (Claude-review follow-up).
#
# Question: is the emergent channel width, measured in metres, invariant
# under grid coarsening (the pattern is physics-set and coarsening is
# viable up to ~width/4), or does it scale with dX (the pattern is
# grid-controlled and coarsening is meaningless)?
#
# Design (fixed by the stability review):
#   - Same physical domain 1024x512 m for every resolution: the grid is
#     (int(1024/dx) x int(512/dx)) cells, always powers of 2 like the CLI.
#   - dT stays 0.0125 s and EndTime stays 70580 (morphological) s: the
#     binding friction limit has no dX dependence, so all resolutions are
#     equally stable and equally evolved. Cost then scales exactly with the
#     cell count.
#   - Grid Pe = |U|*dX/DifU stays 0.15-0.6 for dx in 0.25-1.0 m, inside the
#     DifU-dominated window where the centred advective scheme is safe
#     (2 m would sit at the edge; run it only if the trend needs it).
#   - No sfere/model.py changes: results stay directly comparable to RUN9.
#
# Subcommands:
#   smoke            1 short frame per resolution: s/step, memory, NaN check,
#                    and the extrapolated full-run wall time (before you commit
#                    the GPU for hours).
#   run --dx 0.25    one full run (EndTime 70580, 20 frames), checkpoints every
#                    2 frames; --resume continues from the last checkpoint
#                    (note: the saved per-frame stacks then cover only the
#                    post-resume frames; the final arrays are always complete).
#   report           widths in metres for every available run, calibrated on
#                    RUN9 first (--calibrate-only); writes figures + verdict.
#
# Outputs land in DIAGNOSTIC/ (npy/npz are git-ignored; figures + verdict
# are committed).

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sfere import width
from sfere.config import GradientOptions, GridConfig, PhysParams, RunConfig, SimConfig
from sfere.io import save_all
from sfere.model import Params
from sfere.simulate import Simulation, stack
from sfere.state import init_conditions
from sfere.viz import save_figure

DOMAIN_X_M = 1024.0  # [m] physical domain (LengthX of the RUN9 setup)
DOMAIN_Y_M = 512.0   # [m] physical domain (LengthY of the RUN9 setup)
END_TIME = 70580.0   # [hydrodynamic s] as in RUN9
D_T = 0.0125         # [s] as in RUN9: friction-limited, dX-independent
NUM_FRAMES = 20
CHECKPOINT_EVERY = 2
OUT_DIR = "DIAGNOSTIC"
RUN9_DIR = "RUN9"


# --- shared setup -----------------------------------------------------------

def _grid_for(dx: float) -> GridConfig:
    w, h = int(round(DOMAIN_X_M / dx)), int(round(DOMAIN_Y_M / dx))
    for v in (w, h):
        if v & (v - 1) or v < 16:
            raise SystemExit(f"dx={dx} gives grid cell count {v}, not a power of 2 >= 16")
    return GridConfig(Block_Size_X=16, Block_Size_Y=16,
                      Block_Number_X=w // 16, Block_Number_Y=h // 16,
                      dX=dx, dY=dx)


def _bundle(dx: float, endtime: float, num_frames: int):
    """(grid, params, sim, run, steps) for one resolution, RUN9 parameters."""
    grid = _grid_for(dx)
    phys = PhysParams()
    sim = SimConfig(EndTime=endtime, NumFrames=num_frames, dT=D_T)
    outdir = f"{OUT_DIR}/dx{dx:g}" if endtime == END_TIME else f"{OUT_DIR}/smoke_dx{dx:g}"
    run = RunConfig(outdir=outdir)
    params = Params(phys=phys, gradients=GradientOptions(), dX=grid.dX, dY=grid.dY, dT=sim.dT)
    return grid, params, sim, run, sim.steps_per_frame


def _load_base(outdir: str) -> str:
    return f"{outdir}/ComplexChannelPatterns_Fig1f_RUN9_SUBRUNa1_SUBRUNb1"


def _load_final(outdir: str, name: str) -> np.ndarray:
    return np.load(f"{_load_base(outdir)}_{name}.npy")


def _has_final(outdir: str) -> bool:
    return os.path.exists(f"{_load_base(outdir)}_u.npy")


# --- smoke --------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> None:
    import jax

    print(f"backend: {jax.default_backend()} devices: {jax.devices()}")
    print(f"{'dx':>5} {'grid':>11} {'steps':>8} {'s/frame':>9} {'s/step':>9} "
          f"{'mem [GB]':>8} {'ok':>5}  full-run estimate")
    for dx in args.dx:
        grid, params, sim, run, steps = _bundle(dx, 70.58, 1)  # one short frame
        state = init_conditions(phys=params.phys, grid=grid, seed=2)
        driver = Simulation(state, params, sim)
        driver.advance(1)
        jax.block_until_ready(driver.state.h)
        per_frame = driver.times[0]
        s_per_step = per_frame / steps
        try:
            mem = jax.devices()[0].memory_stats().get("bytes_in_use", 0) / 2**30
        except Exception:
            mem = float("nan")
        ok = all(np.isfinite(driver.latest_frame()[n]).all()
                 for n in ("u", "v", "h", "s", "d"))
        _, _, _, _, steps_full = _bundle(dx, END_TIME, NUM_FRAMES)
        full_h = s_per_step * steps_full * NUM_FRAMES / 3600
        est = f"~{full_h:.1f} h" if ok else "(NaN in state!)"
        print(f"{dx:>5} {grid.Grid_Width}x{grid.Grid_Height:<6} {steps:>8} "
              f"{per_frame:>9.1f} {s_per_step:>9.4f} {mem:>8.2f} {str(ok):>5}  {est}")


# --- run ----------------------------------------------------------------------

def _resume_state(outdir: str):
    """(state arrays dict, completed frame index) from the last checkpoint."""
    import glob

    checkpoints = sorted(glob.glob(f"{outdir}/checkpoint_*.npz"),
                         key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not checkpoints:
        return None, -1
    with np.load(checkpoints[-1]) as z:
        frame = int(z["frame"])
        arrays = {k: np.asarray(z[k]) for k in z.files if k != "frame"}
    return arrays, frame


def cmd_run(args: argparse.Namespace) -> None:
    dx = args.dx
    grid, params, sim, run, steps = _bundle(dx, END_TIME, NUM_FRAMES)
    os.makedirs(run.outdir, exist_ok=True)

    state0 = init_conditions(phys=params.phys, grid=grid, vegetation=True, seed=2)
    start_frame = -1
    if args.resume:
        arrays, start_frame = _resume_state(run.outdir)
        if arrays is not None:
            import jax.numpy as jnp

            state0 = dataclasses.replace(
                state0, **{k: jnp.asarray(v) for k, v in arrays.items()})
            print(f"resuming from frame checkpoint {start_frame}; saved stacks will "
                  f"cover only the resume tail")
        else:
            print("no checkpoint found; starting fresh")

    driver = Simulation(state0, params, sim)
    t0 = time.time()
    t_prev = t0
    for frame in range(start_frame + 1, NUM_FRAMES):
        driver.advance(1)
        t_now = time.time()
        eta = (t_now - t_prev) * (NUM_FRAMES - 1 - frame)
        print(f"frame {frame + 1}/{NUM_FRAMES}: {t_now - t_prev:.1f} s, "
              f"elapsed {(t_now - t0) / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)
        t_prev = t_now
        if (frame + 1) % CHECKPOINT_EVERY == 0 or frame + 1 == NUM_FRAMES:
            latest = {n: np.asarray(getattr(driver.state, n)) for n in
                      ("u", "v", "h", "b", "s", "d", "c", "t")}
            np.savez(f"{run.outdir}/checkpoint_{frame}.npz", frame=frame, **latest)

    stacks = stack(driver.frames)
    save_all(stacks, run)
    print(f"figure: {save_figure(stacks, run)}")
    print(f"done: {run.outdir} in {(time.time() - t0) / 3600:.2f} h")


# --- report -------------------------------------------------------------------

def _fields_for(outdir: str, phys: PhysParams) -> dict[str, np.ndarray]:
    """Channel fields (unit-width discharge |u|*h, recomputed shear t)."""
    u = _load_final(outdir, "u")
    v = _load_final(outdir, "v")
    h = _load_final(outdir, "h")
    d = _load_final(outdir, "d")
    return {"q": np.hypot(u, v) * h,
            "t": width.shear(u, v, h, d, nb=phys.nb, nv=phys.nv, kk=phys.kk,
                             rho=phys.rho, g=phys.g)}


def cmd_report(args: argparse.Namespace) -> None:
    phys = PhysParams()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.calibrate_only:
        if not _has_final(RUN9_DIR):
            raise SystemExit("no RUN9 output found to calibrate on")
        fields = _fields_for(RUN9_DIR, phys)
        results = {0.5: {name: width.measure_field(fields[name], 0.5)
                         for name in fields}}
        print(f"{RUN9_DIR} (dx=0.5 m, calibration) done")
        print(_calibration_figure(fields, results, args.mask_quantile))
        return

    # Fresh runs where they exist; RUN9 stands in for dx=0.5 otherwise.
    measured: dict[float, dict[str, dict]] = {}
    fields_by_dx: dict[float, dict[str, np.ndarray]] = {}
    for dx in (1.0, 0.5, 0.25):
        d = f"{OUT_DIR}/dx{dx:g}"
        if _has_final(d):
            pass
        elif dx == 0.5 and _has_final(RUN9_DIR):
            d = RUN9_DIR
        else:
            print(f"dx={dx}: no output in {OUT_DIR}, skipped")
            continue
        fields = _fields_for(d, phys)
        fields_by_dx[dx] = fields
        measured[dx] = {name: width.measure_field(fields[name], dx)
                        for name in fields}
        print(f"{d} (dx={dx} m) done")
    if not measured:
        raise SystemExit(f"no runs found under {OUT_DIR} (run them first, or use "
                         "--calibrate-only)")
    dxs = sorted(measured)

    # Port regression check: fresh dx=0.5 run vs RUN9, bit-comparison.
    regression = None
    if _has_final(f"{OUT_DIR}/dx0.5") and _has_final(RUN9_DIR):
        maxima = {n: float(np.max(np.abs(_load_final(RUN9_DIR, n).astype(np.float64)
                                         - _load_final(f"{OUT_DIR}/dx0.5", n)
                                         .astype(np.float64))))
                  for n in ("u", "v", "h", "s", "d")}
        regression = maxima
        print("dx=0.5 vs RUN9 max |delta| per field:",
              ", ".join(f"{n}: {v:.3e}" for n, v in maxima.items()))

    print(_calibration_figure(fields_by_dx.get(0.5) or _fields_for(RUN9_DIR, phys),
                              measured, args.mask_quantile))
    print(_scale_figure(fields_by_dx, measured, args.mask_quantile))
    print(_verdict_markdown(measured, regression))


def _calibration_figure(fields: dict, results: dict, mask_quantile: float) -> str:
    """Width vs threshold, per estimator: is the width number robust?"""
    dx = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    field_names = [k for k in fields]
    qs = sorted({q for name in field_names
                 for q in results[dx][name] if q.startswith("q")},
                key=lambda k: float(k[1:]))
    for field_name in field_names:
        for metric, label in (("runs_median", "runs median"), ("edt", "EDT")):
            ys = [results[dx][field_name][q][metric] for q in qs]
            ax.plot([float(q[1:]) for q in qs], ys, marker="o",
                    label=f"{label} ({field_name})")
    spectrum = results[dx][field_names[0]]["spectrum"]
    ax.axhline(spectrum, color="gray", ls=":",
               label=f"spectrum spacing ({field_names[0]}-field): {spectrum:.1f} m")
    ax.set(xlabel="threshold quantile", ylabel="width [m]",
           title=f"calibration on the {'/'.join(field_names)} field")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    q_field = _channels(fields)
    mask = width.channel_mask(q_field, mask_quantile)
    widths = width.runs_width(mask, dx)
    ax.imshow(q_field, origin="upper", aspect="auto", cmap="viridis")
    ax.contour(mask.astype(float), levels=[0.5], colors="w", linewidths=0.8)
    ax.set(title=f"|u|*h, mask at q={mask_quantile} "
                 f"(median width {np.median(widths):.1f} m)")
    path = f"{OUT_DIR}/width_calibration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


def _channels(fields: dict[str, np.ndarray]) -> np.ndarray:
    """The primary channel field stored under the 'q' key."""
    return fields["q"]


def _scale_figure(fields_by_dx: dict, measured: dict, mask_quantile: float) -> str:
    dxs = sorted(measured)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for field_name in ("q", "t"):
        if field_name not in measured[dxs[0]]:
            continue
        for metric, label in (("runs_median", "runs median"), ("edt", "EDT")):
            ys = [measured[dx][field_name]["q0.95"][metric] for dx in dxs]
            ax.plot(dxs, ys, marker="o", label=f"{label} ({field_name})")
    ax.plot(dxs, [measured[dx]["q"]["spectrum"] for dx in dxs], ":", marker="o",
            color="gray", label="spectrum spacing (q-field)")
    ax.set(xscale="log", yscale="log", xlabel="dX [m]", ylabel="width [m]",
           title="channel width vs resolution (q0.95)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for dx in dxs:
        mask = width.channel_mask(_channels(fields_by_dx[dx]), mask_quantile)
        widths = width.runs_width(mask, dx)
        ax.hist(widths, bins=np.logspace(-0.7, np.log10(max(widths.max(), 1.0)), 40),
                histtype="step", density=True,
                label=f"dX={dx} m (median {np.median(widths):.1f})")
    ax.set(xscale="log", xlabel="cross-section width [m]", ylabel="pdf",
           title="per-run width distributions (q0.95)")
    ax.legend(fontsize=9)
    path = f"{OUT_DIR}/width_vs_dx.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


def _fmt(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.2f}"


def _verdict_markdown(measured: dict, regression) -> str:
    dxs = sorted(measured)
    baseline = measured[0.5]["q"]["q0.95"]["runs_median"]
    lines = [
        "# dX-invariance verdict (channel width in metres)",
        "",
        "Primary estimator: runs median on the |u|*h field at q0.95.",
        "",
        "| dX [m] | runs median [m] | EDT [m] | spectrum spacing [m] | ratio vs 0.5 m |",
        "|---|---|---|---|---|",
    ]
    for dx in dxs:
        r = measured[dx]["q"]["q0.95"]
        ratio = r["runs_median"] / baseline if baseline else float("nan")
        lines.append(f"| {dx} | {_fmt(r['runs_median'])} | {_fmt(r['edt'])} "
                     f"| {_fmt(measured[dx]['q']['spectrum'])} | {ratio:.2f} |")
    lines.append("")
    lines.append("Cross-check, t-field (recomputed shear):")
    for dx in dxs:
        r = measured[dx]["t"]["q0.95"]
        lines.append(f"- dX={dx}: runs median {_fmt(r['runs_median'])} m, "
                     f"EDT {_fmt(r['edt'])} m")
    primary = [measured[dx]["q"]["q0.95"]["runs_median"] for dx in dxs]
    if len(dxs) >= 2 and all(np.isfinite(primary)):
        slope = float(np.polyfit(np.log(dxs), np.log(primary), 1)[0])
        lines += [
            "",
            f"Width-scaling exponent (log-log slope of runs median vs dX): **{slope:.2f}**",
            "(physics-set: |slope| <~ 0.1; grid-controlled: slope >~ 0.5)",
            "",
        ]
        if abs(slope) < 0.1:
            lines.append("**Verdict: dX-invariant - the pattern is physics-set; "
                         "coarsening is meaningful (up to ~width/4).**")
        elif slope > 0.5:
            lines.append("**Verdict: grid-controlled - width follows dX; coarsening "
                         "is not meaningful in the current setup.**")
        else:
            lines.append("**Verdict: ambiguous - width partially follows dX; check "
                         "the threshold sensitivity and a resolution-independent IC run.**")
    if regression:
        lines += ["", "Port regression check (fresh dx=0.5 vs RUN9, max |delta|): "
                  + ", ".join(f"{n}: {v:.3e}" for n, v in regression.items())]
    path = f"{OUT_DIR}/verdict.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --- cli ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="dX-invariance diagnostic for the "
                                             "emergent channel width",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="one short frame per resolution: timing/memory/NaN check")
    s.add_argument("--dx", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("run", help=f"one full run (EndTime={END_TIME}, {NUM_FRAMES} frames)")
    r.add_argument("--dx", type=float, required=True, help="cell size [m]")
    r.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="widths per resolution: figures + verdict")
    rep.add_argument("--calibrate-only", action="store_true",
                     help="calibrate the estimators on RUN9 only")
    rep.add_argument("--mask-quantile", type=float, default=0.95,
                     help="quantile for the contour panel (calibration figure)")
    rep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
