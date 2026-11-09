# Configuration dataclasses for the SFERE JAX model.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientOptions:
    """Runtime switches replacing the original's compile-time Gradient_* blocks."""

    D0: bool = False
    Hin: bool = False
    DifD: bool = False
    pD: bool = False


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    """Improvement-layer switches (docs/numerics.md); defaults = the faithful port.

    semi_implicit_friction: the kernel's friction term -g_Ct2*|u|*u/h moves
      from the explicit update into a backward-Euler divisor (1 + dT*lambda)
      with lambda lagged to step-start |u|, as in the kernel. Removes the
      dT < 2/lambda cap (~0.017 s) that vegetated thin films set independently
      of dX.
    upwind_advection: the momentum advection stencil (d_dx/d_dy, centred)
      becomes the flow-direction (upwind) stencil; continuity, sediment and
      vegetation stencils stay exactly as in the kernel. Keeps the scheme
      damped where the grid Peclet number Pe = |U|*dX/DifU approaches 1
      (relevant whenever dX grows toward the coarsening limit).
    """

    semi_implicit_friction: bool = False
    upwind_advection: bool = False


@dataclass(frozen=True)
class PhysParams:
    """Physical/ecological parameters, RUN9 defaults from the original notebook.

    Parameters pre-multiplied by MORF in the original (S_in, E0, Ec, rr, D0,
    DifD) are stored already scaled. Units in comments, as in the original.
    """

    MORF: float = 44712.0

    # Hydrodynamics
    H0: float = 0.02        # [m] imposed initial water layer thickness
    g: float = 9.81         # [m/s^2]
    HCrit: float = 0.001    # [m] critical water layer thickness
    slope: float = 0.000    # [m/m] uniform subsediment bed slope
    DifU: float = 5e-1      # [m^2/s] turbulent eddy viscosity
    rho: float = 1000.0     # [kg/m3]

    # Vegetation roughness
    nb: float = 0.016       # [s m^(-1/3)] Manning roughness, bare sediment
    nv: float = 0.2         # [s m^(-1/3)] Manning roughness, vegetated
    Cb: float = (0.02 ** (1 / 6)) / 0.016  # [m^(1/2)/s] Chezy, initial condition

    # Sedimentation
    S_in: float = 44712.0 * 5e-9   # [s/tide]*[m/s] max sediment input rate
    Qs: float = 6e-4               # [m] water thickness at half sediment input
    E0: float = 44712.0 * 2.5e-4   # [s/tide]*[s m^-2] erosion rate (bare, unit shear)
    pE: float = 0.9                # [-] erosion reduction at veg. carrying capacity

    # Vegetation
    rr: float = 44712.0 * 3.2e-8   # [s/tide]*[1/s] intrinsic growth rate
    kk: float = 1500.0             # [m^-2] carrying capacity
    Qq: float = 2e-2               # [m] water thickness at half growth
    Ec: float = 44712.0 * 1e-5     # [s/tide]*[s m^-2] vegetation erosion rate
    p_est: float = 0.002           # [-] seedling establishment probability

    # Gradients (log-interpolated across width when the switch is on)
    D0: float = 44712.0 * 1e-7
    D0min: float = 44712.0 * 1e-7 / 2.0
    D0max: float = 44712.0 * 1e-7 * 200.0
    Hin: float = 1.0e-5    # [m/s] tidally averaged discharge velocity
    HinMin: float = 1.0e-5
    HinMax: float = 1.0e-4
    DifD: float = 44712.0 * 6e-9
    DifDmin: float = 44712.0 * 6e-9 / 10.0
    DifDmax: float = 44712.0 * 6e-9 * 10.0
    pD: float = 0.99
    pDmin: float = 0.099
    pDmax: float = 9.9

    # Infiltration
    Df: float = 0.000      # [1/s] vertical drainage rate

    @property
    def facTY(self) -> float:
        """[yr/tide] tides-to-years conversion."""
        return self.MORF / (60 * 60 * 24 * 365.25)


@dataclass(frozen=True)
class GridConfig:
    """Spatial grid; powers-of-2 block layout mirrors the original."""

    Block_Size_X: int = 16
    Block_Size_Y: int = 16
    Block_Number_X: int = 128
    Block_Number_Y: int = 64
    dX: float = 0.5        # [m]
    dY: float = 0.5        # [m]

    @property
    def Grid_Width(self) -> int:
        return self.Block_Size_X * self.Block_Number_X

    @property
    def Grid_Height(self) -> int:
        return self.Block_Size_Y * self.Block_Number_Y

    @property
    def LengthX(self) -> float:
        return self.Grid_Width * self.dX

    @property
    def LengthY(self) -> float:
        return self.Grid_Height * self.dY


@dataclass(frozen=True)
class SimConfig:
    """Time integration settings (RUN9 defaults)."""

    EndTime: float = 70580.0   # [tide] = hydrodynamic seconds via MORF
    NumFrames: int = 100
    dT: float = 0.0125         # [s]

    @property
    def steps_per_frame(self) -> int:
        return int(self.EndTime / self.dT / self.NumFrames)


@dataclass(frozen=True)
class RunConfig:
    """Run identity, mirroring ModelName/RUN/SUBRUN bookkeeping."""

    RUN: int = 9
    SUBRUNa: int = 1
    SUBRUNb: int = 1
    outdir: str = "RUN9"
    seed: int = 2
    vegetation: bool = True
    tag: str = ""  # labels deviation runs, e.g. "IMPf" (see docs/numerics.md)

    @property
    def ModelName(self) -> str:
        base = f"ComplexChannelPatterns_Fig1f_RUN{self.RUN}"
        return f"{base}_{self.tag}" if self.tag else base

    def replace(self, **kw) -> "RunConfig":
        return dataclasses.replace(self, **kw)