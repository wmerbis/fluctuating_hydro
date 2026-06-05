#!/usr/bin/env python3
"""Mass-drift diagnostics for the optimized stochastic FV solver.

This script intentionally does not change the numerical scheme.  It isolates the
solver substeps and prints stage-by-stage mass accounting so drift can be traced
before making a targeted solver fix.

Current diagnostic expectations:
* explicit finite-volume RHS: conserves M_A, M_B, M_occ, and M_total to roundoff;
* semi-implicit Gamma step: should leave the rfft2 k=(0, 0) mode unchanged and
  conserve M_A, M_B, M_occ, and M_total to roundoff;
* stochastic increment: conservative Schelling face noise conserves species
  masses, voter noise changes M_A and M_B oppositely, and the combined stochastic
  increment conserves M_occ and M_total to roundoff;
* simplex/positivity projection is local and non-conservative by design when it
  clips negative densities, clips inactive cells, or rescales occupied density
  above one.  Projection drift is reported separately, not treated as an
  unintended solver drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_volume_stochastic import (
    ModelParameters,
    SchellingVoterFVSolver,
    _compute_rho0_inplace,
    _compute_rho0_periodic_inplace,
    _stochastic_increment_numba,
    _stochastic_increment_periodic_numba,
    make_random_initial_condition,
)

Array = np.ndarray
Masses = Dict[str, float]


@dataclass
class ProjectionStepStats:
    """Mass and cell accounting for all projection calls in one time step."""

    events: int = 0
    changed_cells: int = 0
    negative_A_cells: int = 0
    negative_B_cells: int = 0
    occupied_above_one_cells: int = 0
    dM_A: float = 0.0
    dM_B: float = 0.0
    dM_occ: float = 0.0
    dM_total: float = 0.0
    abs_dM_A: float = 0.0
    abs_dM_B: float = 0.0
    abs_dM_occ: float = 0.0
    abs_dM_total: float = 0.0


@dataclass
class LongRunResult:
    name: str
    projection_enabled: bool
    dt: float
    nsteps: int
    record_every: int
    times: Array
    steps: Array
    masses: Dict[str, Array]
    drifts: Dict[str, Array]
    projection_events: int
    projection_changed_cells: int
    projection_negative_A_cells: int
    projection_negative_B_cells: int
    projection_occupied_above_one_cells: int
    projection_drift: Masses
    projection_abs_drift: Masses
    drift_stats: Dict[str, Dict[str, float]]
    csv_path: Optional[Path]
    projection_csv_path: Optional[Path]

@dataclass
class CaseResult:
    name: str
    initial: Masses
    final: Masses
    max_abs_step_delta: Masses
    warnings: int


def make_params(nx: int, lx: float) -> ModelParameters:
    """Use a moderately active parameter set without tuning for performance."""
    return ModelParameters(
        D_A=0.1,
        D_B=0.13,
        D_v=0.05,
        beta=8.0,
        h_noise=lx / nx,
        kappa=np.array([[0.6, -0.4], [-0.3, 0.5]], dtype=float),
        gamma=np.array([[1.0, 0.25], [0.15, 0.8]], dtype=float),
    )


def build_solver(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    seed: int,
    *,
    stochastic: bool,
    semi_implicit_stiff: bool,
    simplex_limiter: bool,
    use_periodic_fast_path: bool,
    rho_A0: Array,
    rho_B0: Array,
) -> SchellingVoterFVSolver:
    solver = SchellingVoterFVSolver(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params=make_params(nx, lx),
        bc="periodic",
        semi_implicit_stiff=semi_implicit_stiff,
        stochastic=stochastic,
        simplex_limiter=simplex_limiter,
        rng=np.random.default_rng(seed),
        use_periodic_fast_path=use_periodic_fast_path,
    )
    solver.set_state(rho_A0, rho_B0)
    return solver


def masses_for(solver: SchellingVoterFVSolver, rho_A: Array, rho_B: Array) -> Masses:
    rho0 = np.zeros_like(rho_A)
    _compute_rho0_inplace(rho_A, rho_B, solver.active, rho0)
    active = solver.active
    occ = rho_A + rho_B
    return {
        "A": float(np.sum(rho_A[active]) * solver.vol),
        "B": float(np.sum(rho_B[active]) * solver.vol),
        "occupied": float(np.sum(occ[active]) * solver.vol),
        "vacancy": float(np.sum(rho0[active]) * solver.vol),
        "total": float(np.sum((occ + rho0)[active]) * solver.vol),
    }


def mass_delta(after: Masses, before: Masses) -> Masses:
    return {key: after[key] - before[key] for key in before}


def max_abs_mass(a: Masses, b: Masses) -> Masses:
    return {key: max(abs(a.get(key, 0.0)), abs(b.get(key, 0.0))) for key in set(a) | set(b)}


def print_masses(label: str, masses: Masses, reference: Masses | None = None) -> None:
    if reference is None:
        delta = {key: 0.0 for key in masses}
    else:
        delta = mass_delta(masses, reference)
    print(
        f"  {label:<30} "
        f"M_A={masses['A']:+.16e} (d={delta['A']:+.3e})  "
        f"M_B={masses['B']:+.16e} (d={delta['B']:+.3e})  "
        f"M_occ={masses['occupied']:+.16e} (d={delta['occupied']:+.3e})  "
        f"M_0={masses['vacancy']:+.16e} (d={delta['vacancy']:+.3e})  "
        f"M_total={masses['total']:+.16e} (d={delta['total']:+.3e})"
    )


def warn_if(label: str, delta: Masses, keys: Iterable[str], tol: float) -> int:
    warnings = 0
    for key in keys:
        if abs(delta[key]) > tol:
            print(f"    WARNING {label}: |dM_{key}|={abs(delta[key]):.3e} > {tol:.3e}")
            warnings += 1
    return warnings


def projection_breakdown(before_A: Array, before_B: Array, after_A: Array, after_B: Array) -> Dict[str, int | float]:
    finite_mask = np.isfinite(before_A) & np.isfinite(before_B)
    clipped_nonfinite = int(np.size(before_A) - np.count_nonzero(finite_mask))
    clipped_A_negative = int(np.count_nonzero(np.where(np.isfinite(before_A), before_A, 0.0) < 0.0))
    clipped_B_negative = int(np.count_nonzero(np.where(np.isfinite(before_B), before_B, 0.0) < 0.0))
    clipped_A_above_one = int(np.count_nonzero(before_A > 1.0))
    clipped_B_above_one = int(np.count_nonzero(before_B > 1.0))
    occupied_after_negative_clip = np.maximum(np.where(np.isfinite(before_A), before_A, 0.0), 0.0)
    occupied_after_negative_clip += np.maximum(np.where(np.isfinite(before_B), before_B, 0.0), 0.0)
    rescaled_occupied = int(np.count_nonzero(occupied_after_negative_clip > 1.0))
    changed = int(np.count_nonzero((after_A != before_A) | (after_B != before_B)))
    return {
        "changed_cells": changed,
        "nonfinite_cells": clipped_nonfinite,
        "A_negative_cells": clipped_A_negative,
        "B_negative_cells": clipped_B_negative,
        "A_above_one_cells": clipped_A_above_one,
        "B_above_one_cells": clipped_B_above_one,
        "occupied_above_one_after_negative_clip_cells": rescaled_occupied,
        "max_abs_A_change": float(np.max(np.abs(after_A - before_A))),
        "max_abs_B_change": float(np.max(np.abs(after_B - before_B))),
    }


def run_repeated_case(
    name: str,
    solver: SchellingVoterFVSolver,
    steps: int,
    tol: float,
    conserved_keys: Tuple[str, ...],
    stepper: Callable[[SchellingVoterFVSolver], None],
) -> CaseResult:
    print(f"\n=== {name} ===")
    initial = masses_for(solver, solver.rho_A, solver.rho_B)
    print_masses("initial", initial)
    max_step_delta = {key: 0.0 for key in initial}
    warnings = 0
    previous = initial
    for n in range(1, steps + 1):
        stepper(solver)
        current = masses_for(solver, solver.rho_A, solver.rho_B)
        step_delta = mass_delta(current, previous)
        max_step_delta = max_abs_mass(max_step_delta, step_delta)
        warnings += warn_if(f"step {n}", step_delta, conserved_keys, tol)
        previous = current
    final = masses_for(solver, solver.rho_A, solver.rho_B)
    print_masses("final", final, initial)
    cumulative = mass_delta(final, initial)
    warnings += warn_if("cumulative", cumulative, conserved_keys, tol)
    print("  max |single-step delta|:", ", ".join(f"{k}={v:.3e}" for k, v in sorted(max_step_delta.items())))
    return CaseResult(name, initial, final, max_step_delta, warnings)


def explicit_stepper(dt: float) -> Callable[[SchellingVoterFVSolver], None]:
    def _step(solver: SchellingVoterFVSolver) -> None:
        rho0 = solver._compute_rho0_for_state(solver.rho_A, solver.rho_B)
        rhs_A, rhs_B = solver.explicit_rhs(solver.rho_A, solver.rho_B, rho0=rho0)
        solver.rho_A[...] = solver.rho_A + dt * rhs_A
        solver.rho_B[...] = solver.rho_B + dt * rhs_B
    return _step


def stochastic_stepper(dt: float) -> Callable[[SchellingVoterFVSolver], None]:
    def _step(solver: SchellingVoterFVSolver) -> None:
        rho0 = solver._compute_rho0_for_state(solver.rho_A, solver.rho_B)
        dW_A, dW_B = solver.stochastic_increment(solver.rho_A, solver.rho_B, dt, rho0=rho0)
        solver.rho_A[...] = solver.rho_A + dW_A
        solver.rho_B[...] = solver.rho_B + dW_B
    return _step


def gamma_stepper(dt: float) -> Callable[[SchellingVoterFVSolver], None]:
    def _step(solver: SchellingVoterFVSolver) -> None:
        before_hat_A00 = np.fft.rfft2(solver.rho_A)[0, 0]
        before_hat_B00 = np.fft.rfft2(solver.rho_B)[0, 0]
        rho_A_new, rho_B_new = solver.semi_implicit_gamma_step(solver.rho_A, solver.rho_B, dt)
        after_hat_A00 = np.fft.rfft2(rho_A_new)[0, 0]
        after_hat_B00 = np.fft.rfft2(rho_B_new)[0, 0]
        zero_mode_error = max(abs(after_hat_A00 - before_hat_A00), abs(after_hat_B00 - before_hat_B00))
        if zero_mode_error > 1.0e-10:
            print(f"    WARNING gamma zero-mode changed by {zero_mode_error:.3e}")
        solver.rho_A[...] = rho_A_new
        solver.rho_B[...] = rho_B_new
    return _step


def projection_stepper() -> Callable[[SchellingVoterFVSolver], None]:
    def _step(solver: SchellingVoterFVSolver) -> None:
        before_A = solver.rho_A.copy()
        before_B = solver.rho_B.copy()
        before = masses_for(solver, before_A, before_B)
        corrections = solver.apply_simplex_limiter()
        after = masses_for(solver, solver.rho_A, solver.rho_B)
        delta = mass_delta(after, before)
        breakdown = projection_breakdown(before_A, before_B, solver.rho_A, solver.rho_B)
        print(
            f"    projection corrections={corrections}, "
            f"dM_A={delta['A']:+.3e}, dM_B={delta['B']:+.3e}, "
            f"dM_occ={delta['occupied']:+.3e}, changed_cells={breakdown['changed_cells']}, "
            f"negA={breakdown['A_negative_cells']}, negB={breakdown['B_negative_cells']}, "
            f"occ>1={breakdown['occupied_above_one_after_negative_clip_cells']}"
        )
    return _step


def combined_step_accounting(solver: SchellingVoterFVSolver, dt: float, add_noise: bool, tol: float) -> int:
    """Run one full solver-order step and print mass after every substep."""
    warnings = 0
    before = masses_for(solver, solver.rho_A, solver.rho_B)
    print_masses("before update", before)

    rho0 = solver._compute_rho0_for_state(solver.rho_A, solver.rho_B)
    rhs_A, rhs_B = solver.explicit_rhs(solver.rho_A, solver.rho_B, rho0=rho0)
    solver._rho_A_star[...] = solver.rho_A + dt * rhs_A
    solver._rho_B_star[...] = solver.rho_B + dt * rhs_B
    after_explicit = masses_for(solver, solver._rho_A_star, solver._rho_B_star)
    print_masses("after explicit FV", after_explicit, before)
    warnings += warn_if("explicit", mass_delta(after_explicit, before), ("A", "B", "occupied", "total"), tol)

    if add_noise:
        dW_A, dW_B = solver.stochastic_increment(solver.rho_A, solver.rho_B, dt, rho0=rho0)
        solver._rho_A_star += dW_A
        solver._rho_B_star += dW_B
    after_stochastic = masses_for(solver, solver._rho_A_star, solver._rho_B_star)
    print_masses("after stochastic", after_stochastic, after_explicit)
    if add_noise:
        warnings += warn_if("stochastic", mass_delta(after_stochastic, after_explicit), ("occupied", "total"), tol)

    if solver.simplex_limiter:
        pre_projection_A = solver._rho_A_star.copy()
        pre_projection_B = solver._rho_B_star.copy()
        corrections = solver.apply_simplex_limiter(solver._rho_A_star, solver._rho_B_star)
        after_pre_gamma_projection = masses_for(solver, solver._rho_A_star, solver._rho_B_star)
        print_masses(f"after pre-gamma projection ({corrections} cells)", after_pre_gamma_projection, after_stochastic)
        print("    projection breakdown:", projection_breakdown(pre_projection_A, pre_projection_B, solver._rho_A_star, solver._rho_B_star))
    else:
        after_pre_gamma_projection = after_stochastic

    if solver.semi_implicit_stiff:
        zero_before_A = np.fft.rfft2(solver._rho_A_star)[0, 0]
        zero_before_B = np.fft.rfft2(solver._rho_B_star)[0, 0]
        rho_A_new, rho_B_new = solver.semi_implicit_gamma_step(solver._rho_A_star, solver._rho_B_star, dt)
        zero_after_A = np.fft.rfft2(rho_A_new)[0, 0]
        zero_after_B = np.fft.rfft2(rho_B_new)[0, 0]
        print(f"    gamma zero-mode delta: A={zero_after_A - zero_before_A:+.3e}, B={zero_after_B - zero_before_B:+.3e}")
    else:
        rho_A_new = solver._rho_A_star
        rho_B_new = solver._rho_B_star
    after_gamma = masses_for(solver, rho_A_new, rho_B_new)
    print_masses("after Gamma step", after_gamma, after_pre_gamma_projection)
    if solver.semi_implicit_stiff:
        warnings += warn_if("gamma", mass_delta(after_gamma, after_pre_gamma_projection), ("A", "B", "occupied", "total"), tol)

    solver.rho_A[...] = rho_A_new
    solver.rho_B[...] = rho_B_new
    if solver.simplex_limiter:
        pre_projection_A = solver.rho_A.copy()
        pre_projection_B = solver.rho_B.copy()
        corrections = solver.apply_simplex_limiter()
        after_final_projection = masses_for(solver, solver.rho_A, solver.rho_B)
        print_masses(f"after final projection ({corrections} cells)", after_final_projection, after_gamma)
        print("    projection breakdown:", projection_breakdown(pre_projection_A, pre_projection_B, solver.rho_A, solver.rho_B))
    return warnings


def compare_fast_and_generic_stochastic(
    rho_A0: Array,
    rho_B0: Array,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    dt: float,
    seed: int,
) -> int:
    print("\n=== stochastic generic-periodic vs fast-periodic kernel agreement ===")
    generic = build_solver(nx, ny, lx, ly, seed, stochastic=True, semi_implicit_stiff=False, simplex_limiter=False,
                           use_periodic_fast_path=False, rho_A0=rho_A0, rho_B0=rho_B0)
    fast = build_solver(nx, ny, lx, ly, seed, stochastic=True, semi_implicit_stiff=False, simplex_limiter=False,
                        use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0)
    rng = np.random.default_rng(seed + 1000)
    eta_A_x = rng.standard_normal((ny, nx))
    eta_B_x = rng.standard_normal((ny, nx))
    eta_A_y = rng.standard_normal((ny, nx))
    eta_B_y = rng.standard_normal((ny, nx))
    xi = rng.standard_normal((ny, nx))
    rho0_generic = np.zeros_like(rho_A0)
    rho0_fast = np.zeros_like(rho_A0)
    _compute_rho0_inplace(generic.rho_A, generic.rho_B, generic.active, rho0_generic)
    _compute_rho0_periodic_inplace(fast.rho_A, fast.rho_B, rho0_fast)
    _stochastic_increment_numba(
        generic.rho_A, generic.rho_B, generic.active, generic.dx, generic.dy, generic.vol, generic.bc_periodic,
        generic.params.D_A, generic.params.D_B, generic.params.D_v, generic.params.h_noise,
        generic.params.spatial_dim, dt, rho0_generic, generic._dW_A, generic._dW_B,
        eta_A_x, eta_B_x, eta_A_y, eta_B_y, xi,
    )
    _stochastic_increment_periodic_numba(
        fast.rho_A, fast.rho_B, fast.dx, fast.dy, fast.vol,
        fast.params.D_A, fast.params.D_B, fast.params.D_v, fast.params.h_noise,
        fast.params.spatial_dim, dt, rho0_fast, fast._dW_A, fast._dW_B,
        eta_A_x, eta_B_x, eta_A_y, eta_B_y, xi,
    )
    max_A = float(np.max(np.abs(generic._dW_A - fast._dW_A)))
    max_B = float(np.max(np.abs(generic._dW_B - fast._dW_B)))
    sum_occ = float(abs(np.sum(generic._dW_A + generic._dW_B) * generic.vol))
    print(f"  max |dW_A_generic - dW_A_fast| = {max_A:.3e}")
    print(f"  max |dW_B_generic - dW_B_fast| = {max_B:.3e}")
    print(f"  generic stochastic occupied-increment mass = {sum_occ:.3e}")
    if max(max_A, max_B) > 0.0:
        print("    WARNING kernels differ despite identical random buffers")
        return 1
    return 0


def user_long_run_params(nx: int, ny: int, lx: float, ly: float) -> ModelParameters:
    """Parameter set requested for the long-run mass-drift stress diagnostic."""
    return ModelParameters(
        D_A=0.1,
        D_B=0.1,
        D_v=0.0,
        beta=10.0,
        h_noise=min(lx / nx, ly / ny),
        kappa=np.array([[0.6, -0.4], [-0.4, 0.6]], dtype=float),
        gamma=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )


def clone_params(params: ModelParameters, **overrides: Any) -> ModelParameters:
    values = {
        "D_A": params.D_A,
        "D_B": params.D_B,
        "D_v": params.D_v,
        "beta": params.beta,
        "h_noise": params.h_noise,
        "spatial_dim": params.spatial_dim,
        "kappa": params.kappa.copy(),
        "gamma": params.gamma.copy(),
    }
    values.update(overrides)
    return ModelParameters(**values)


def build_long_run_solver(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    seed: int,
    params: ModelParameters,
    projection_enabled: bool,
    rho_A0: Array,
    rho_B0: Array,
) -> SchellingVoterFVSolver:
    solver = SchellingVoterFVSolver(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params=params,
        bc="periodic",
        semi_implicit_stiff=True,
        stochastic=True,
        simplex_limiter=projection_enabled,
        rng=np.random.default_rng(seed),
        use_periodic_fast_path=True,
    )
    solver.set_state(rho_A0, rho_B0)
    return solver


def make_sharp_interface_initial_condition(nx: int, ny: int, seed: int, total_occ: float = 0.92) -> Tuple[Array, Array]:
    """Occupied, segregated state used to stress sharp interfaces without changing the scheme."""
    rng = np.random.default_rng(seed)
    rho_A = np.empty((ny, nx), dtype=float)
    rho_B = np.empty((ny, nx), dtype=float)
    left = np.arange(nx)[None, :] < nx // 2
    rho_A[...] = np.where(left, 0.86 * total_occ, 0.08 * total_occ)
    rho_B[...] = np.where(left, 0.08 * total_occ, 0.86 * total_occ)
    perturb = 2.0e-3 * rng.standard_normal((ny, nx))
    rho_A += perturb
    rho_B -= perturb
    np.maximum(rho_A, 0.0, out=rho_A)
    np.maximum(rho_B, 0.0, out=rho_B)
    occ = rho_A + rho_B
    too_full = occ > total_occ
    rho_A[too_full] *= total_occ / occ[too_full]
    rho_B[too_full] *= total_occ / occ[too_full]
    return rho_A, rho_B


def projection_delta_for_arrays(
    solver: SchellingVoterFVSolver,
    before_A: Array,
    before_B: Array,
    after_A: Array,
    after_B: Array,
) -> ProjectionStepStats:
    before = masses_for(solver, before_A, before_B)
    after = masses_for(solver, after_A, after_B)
    delta = mass_delta(after, before)
    breakdown = projection_breakdown(before_A, before_B, after_A, after_B)
    changed = int(breakdown["changed_cells"])
    stats = ProjectionStepStats()
    if changed > 0:
        stats.events = 1
    stats.changed_cells = changed
    stats.negative_A_cells = int(breakdown["A_negative_cells"])
    stats.negative_B_cells = int(breakdown["B_negative_cells"])
    stats.occupied_above_one_cells = int(breakdown["occupied_above_one_after_negative_clip_cells"])
    stats.dM_A = delta["A"]
    stats.dM_B = delta["B"]
    stats.dM_occ = delta["occupied"]
    stats.dM_total = delta["total"]
    stats.abs_dM_A = abs(delta["A"])
    stats.abs_dM_B = abs(delta["B"])
    stats.abs_dM_occ = abs(delta["occupied"])
    stats.abs_dM_total = abs(delta["total"])
    return stats


def add_projection_stats(total: ProjectionStepStats, inc: ProjectionStepStats) -> None:
    for field in (
        "events", "changed_cells", "negative_A_cells", "negative_B_cells", "occupied_above_one_cells",
        "dM_A", "dM_B", "dM_occ", "dM_total", "abs_dM_A", "abs_dM_B", "abs_dM_occ", "abs_dM_total",
    ):
        setattr(total, field, getattr(total, field) + getattr(inc, field))


def diagnostic_full_step(solver: SchellingVoterFVSolver, dt: float, add_noise: bool = True) -> ProjectionStepStats:
    """Mirror ``SchellingVoterFVSolver.step`` while accounting for projection mass changes."""
    rho0 = solver._compute_rho0_for_state(solver.rho_A, solver.rho_B)
    rhs_A, rhs_B = solver.explicit_rhs(solver.rho_A, solver.rho_B, rho0=rho0)
    solver._rho_A_star[...] = solver.rho_A + dt * rhs_A
    solver._rho_B_star[...] = solver.rho_B + dt * rhs_B

    if add_noise:
        dW_A, dW_B = solver.stochastic_increment(solver.rho_A, solver.rho_B, dt, rho0=rho0)
        solver._rho_A_star += dW_A
        solver._rho_B_star += dW_B

    projection_stats = ProjectionStepStats()
    if solver.simplex_limiter:
        before_A = solver._rho_A_star.copy()
        before_B = solver._rho_B_star.copy()
        solver.apply_simplex_limiter(solver._rho_A_star, solver._rho_B_star)
        add_projection_stats(
            projection_stats,
            projection_delta_for_arrays(solver, before_A, before_B, solver._rho_A_star, solver._rho_B_star),
        )

    if solver.semi_implicit_stiff:
        rho_A_new, rho_B_new = solver.semi_implicit_gamma_step(solver._rho_A_star, solver._rho_B_star, dt)
    else:
        rho_A_new = solver._rho_A_star
        rho_B_new = solver._rho_B_star

    if solver.fully_active_periodic:
        solver.rho_A[...] = rho_A_new
        solver.rho_B[...] = rho_B_new
    else:
        solver.rho_A[...] = np.where(solver.active, rho_A_new, 0.0)
        solver.rho_B[...] = np.where(solver.active, rho_B_new, 0.0)

    if solver.simplex_limiter:
        before_A = solver.rho_A.copy()
        before_B = solver.rho_B.copy()
        solver.apply_simplex_limiter()
        add_projection_stats(
            projection_stats,
            projection_delta_for_arrays(solver, before_A, before_B, solver.rho_A, solver.rho_B),
        )
        solver.last_limiter_corrections = projection_stats.changed_cells
    return projection_stats


def drift_series_stats(times: Array, drift: Array) -> Dict[str, float]:
    if len(drift) == 0:
        return {"final": 0.0, "max_abs": 0.0, "rms": 0.0, "slope": 0.0}
    if len(drift) >= 2 and np.ptp(times) > 0.0:
        slope = float(np.polyfit(times, drift, 1)[0])
    else:
        slope = 0.0
    return {
        "final": float(drift[-1]),
        "max_abs": float(np.max(np.abs(drift))),
        "rms": float(np.sqrt(np.mean(drift * drift))),
        "slope": slope,
    }


def write_long_run_csv(path: Path, result: LongRunResult) -> None:
    header = [
        "step", "time", "M_A", "M_B", "M_occ", "M_0", "M_total",
        "dM_A", "dM_B", "dM_occ", "dM_0", "dM_total",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(len(result.steps)):
            row = [
                result.steps[i], result.times[i],
                result.masses["A"][i], result.masses["B"][i], result.masses["occupied"][i],
                result.masses["vacancy"][i], result.masses["total"][i],
                result.drifts["A"][i], result.drifts["B"][i], result.drifts["occupied"][i],
                result.drifts["vacancy"][i], result.drifts["total"][i],
            ]
            fh.write(",".join(str(x) for x in row) + "\n")


def write_projection_activity_csv(path: Path, rows: List[Tuple[int, ProjectionStepStats]]) -> None:
    """Write one row per step so rare projection activations can be located exactly."""
    header = [
        "step", "activated", "changed_cells", "negative_A_cells", "negative_B_cells",
        "occupied_above_one_cells", "dM_A", "dM_B", "dM_occ", "dM_total",
        "abs_dM_A", "abs_dM_B", "abs_dM_occ", "abs_dM_total",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for step, stats in rows:
            row = [
                step,
                int(stats.events > 0),
                stats.changed_cells,
                stats.negative_A_cells,
                stats.negative_B_cells,
                stats.occupied_above_one_cells,
                stats.dM_A,
                stats.dM_B,
                stats.dM_occ,
                stats.dM_total,
                stats.abs_dM_A,
                stats.abs_dM_B,
                stats.abs_dM_occ,
                stats.abs_dM_total,
            ]
            fh.write(",".join(str(x) for x in row) + "\n")


def run_long_run_diagnostic(
    name: str,
    solver: SchellingVoterFVSolver,
    dt: float,
    nsteps: int,
    record_every: int,
    output_dir: Optional[Path] = None,
) -> LongRunResult:
    print(f"\n=== long-run drift stress: {name} ===")
    if record_every <= 0:
        raise ValueError("record_every must be positive")
    initial = masses_for(solver, solver.rho_A, solver.rho_B)
    recorded_steps: List[int] = []
    recorded_times: List[float] = []
    mass_series: Dict[str, List[float]] = {key: [] for key in initial}
    projection_total = ProjectionStepStats()
    projection_rows: List[Tuple[int, ProjectionStepStats]] = []

    def record(step: int) -> None:
        m = masses_for(solver, solver.rho_A, solver.rho_B)
        recorded_steps.append(step)
        recorded_times.append(step * dt)
        for key, value in m.items():
            mass_series[key].append(value)
        delta = mass_delta(m, initial)
        print(
            f"  step {step:>8}/{nsteps}: "
            f"dM_A={delta['A']:+.3e}, dM_B={delta['B']:+.3e}, "
            f"dM_occ={delta['occupied']:+.3e}, dM_0={delta['vacancy']:+.3e}, "
            f"dM_total={delta['total']:+.3e}, projection_events={projection_total.events}"
        )

    record(0)
    for step in range(1, nsteps + 1):
        step_projection = diagnostic_full_step(solver, dt, add_noise=True)
        add_projection_stats(projection_total, step_projection)
        if solver.simplex_limiter:
            projection_rows.append((step, step_projection))
        if step % record_every == 0 or step == nsteps:
            record(step)

    masses = {key: np.array(values, dtype=float) for key, values in mass_series.items()}
    drifts = {key: values - initial[key] for key, values in masses.items()}
    times = np.array(recorded_times, dtype=float)
    steps = np.array(recorded_steps, dtype=int)
    stats = {key: drift_series_stats(times, drift) for key, drift in drifts.items()}
    projection_drift = {
        "A": projection_total.dM_A,
        "B": projection_total.dM_B,
        "occupied": projection_total.dM_occ,
        "vacancy": -projection_total.dM_occ,
        "total": projection_total.dM_total,
    }
    projection_abs_drift = {
        "A": projection_total.abs_dM_A,
        "B": projection_total.abs_dM_B,
        "occupied": projection_total.abs_dM_occ,
        "vacancy": projection_total.abs_dM_occ,
        "total": projection_total.abs_dM_total,
    }
    csv_path = None
    result = LongRunResult(
        name=name,
        projection_enabled=solver.simplex_limiter,
        dt=dt,
        nsteps=nsteps,
        record_every=record_every,
        times=times,
        steps=steps,
        masses=masses,
        drifts=drifts,
        projection_events=projection_total.events,
        projection_changed_cells=projection_total.changed_cells,
        projection_negative_A_cells=projection_total.negative_A_cells,
        projection_negative_B_cells=projection_total.negative_B_cells,
        projection_occupied_above_one_cells=projection_total.occupied_above_one_cells,
        projection_drift=projection_drift,
        projection_abs_drift=projection_abs_drift,
        drift_stats=stats,
        csv_path=None,
        projection_csv_path=None,
    )
    if output_dir is not None:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
        csv_path = output_dir / f"{safe_name}.csv"
        result.csv_path = csv_path
        write_long_run_csv(csv_path, result)
        print(f"  wrote time series: {csv_path}")
        if solver.simplex_limiter:
            projection_csv_path = output_dir / f"{safe_name}.projection_activity.csv"
            result.projection_csv_path = projection_csv_path
            write_projection_activity_csv(projection_csv_path, projection_rows)
            print(f"  wrote projection activity: {projection_csv_path}")

    print_long_run_result(result)
    return result


def print_long_run_result(result: LongRunResult) -> None:
    print("  drift statistics from recorded time series:")
    for key, label in (("A", "M_A"), ("B", "M_B"), ("occupied", "M_occ"), ("vacancy", "M_0"), ("total", "M_total")):
        s = result.drift_stats[key]
        print(
            f"    {label:<7} final={s['final']:+.3e}, max_abs={s['max_abs']:.3e}, "
            f"rms={s['rms']:.3e}, slope={s['slope']:+.3e} per unit time"
        )
    print(
        "  projection summary: "
        f"events={result.projection_events}, changed_cells={result.projection_changed_cells}, "
        f"negA={result.projection_negative_A_cells}, negB={result.projection_negative_B_cells}, "
        f"occ>1={result.projection_occupied_above_one_cells}"
    )
    print(
        "  accumulated projection mass change: "
        f"dM_A={result.projection_drift['A']:+.3e}, dM_B={result.projection_drift['B']:+.3e}, "
        f"dM_occ={result.projection_drift['occupied']:+.3e}, dM_total={result.projection_drift['total']:+.3e}"
    )


def projection_consistency_message(result: LongRunResult, key: str = "occupied") -> str:
    observed = result.drift_stats[key]["final"]
    projected = result.projection_drift[key]
    residual = observed - projected
    scale = max(1.0, abs(observed), abs(projected))
    if abs(residual) <= 1.0e-10 * scale:
        return "consistent with projection corrections to roundoff"
    if abs(projected) > 0.0 and abs(residual) <= 0.05 * max(abs(projected), 1.0e-300):
        return "consistent with projection corrections within 5%"
    return f"not fully explained by projection corrections (residual {residual:+.3e})"


def run_long_run_suite(args: argparse.Namespace) -> int:
    lx = args.lx
    ly = args.ly
    base_params = user_long_run_params(args.nx, args.ny, lx, ly)
    base_A0, base_B0 = make_random_initial_condition(
        args.nx, args.ny, rhoA0=0.31, rhoB0=0.29, noise=2.0e-3, seed=args.seed
    )
    sharp_A0, sharp_B0 = make_sharp_interface_initial_condition(args.nx, args.ny, args.seed + 19)

    weak_params = clone_params(base_params, h_noise=0.5 * base_params.h_noise)
    strong_noise_params = clone_params(base_params, h_noise=2.0 * base_params.h_noise)
    phase_params = clone_params(
        base_params,
        beta=18.0,
        kappa=np.array([[1.1, -0.9], [-0.9, 1.1]], dtype=float),
    )
    trigger_params = clone_params(
        base_params,
        D_A=0.25,
        D_B=0.25,
        beta=24.0,
        h_noise=3.0 * base_params.h_noise,
        kappa=np.array([[1.4, -1.2], [-1.2, 1.4]], dtype=float),
    )
    trigger_A0, trigger_B0 = make_sharp_interface_initial_condition(args.nx, args.ny, args.seed + 23, total_occ=0.985)

    case_specs = [
        ("baseline_projection_disabled", base_params, False, base_A0, base_B0, args.dt),
        ("baseline_projection_enabled", base_params, True, base_A0, base_B0, args.dt),
        ("weak_noise_projection_enabled", weak_params, True, base_A0, base_B0, args.dt),
        ("strong_noise_projection_enabled", strong_noise_params, True, base_A0, base_B0, args.dt),
        ("phase_separating_sharp_projection_enabled", phase_params, True, sharp_A0, sharp_B0, args.dt),
        ("smaller_dt_projection_enabled", base_params, True, base_A0, base_B0, 0.5 * args.dt),
        ("larger_dt_projection_enabled", base_params, True, base_A0, base_B0, 2.0 * args.dt),
        ("projection_triggering_stress", trigger_params, True, trigger_A0, trigger_B0, max(4.0 * args.dt, args.dt)),
    ]
    if args.include_no_projection_variants:
        case_specs.extend([
            ("strong_noise_projection_disabled", strong_noise_params, False, base_A0, base_B0, args.dt),
            ("phase_separating_sharp_projection_disabled", phase_params, False, sharp_A0, sharp_B0, args.dt),
            ("projection_triggering_stress_projection_disabled", trigger_params, False, trigger_A0, trigger_B0, max(4.0 * args.dt, args.dt)),
        ])

    output_dir = Path(args.output_dir) if args.output_dir else None
    results: List[LongRunResult] = []
    for idx, (name, params, projection, rho_A0, rho_B0, dt) in enumerate(case_specs):
        solver = build_long_run_solver(
            args.nx, args.ny, lx, ly, args.seed + 100 * idx, params, projection, rho_A0, rho_B0
        )
        results.append(run_long_run_diagnostic(name, solver, dt, args.nsteps, args.record_every, output_dir))

    print("\n=== long-run diagnostic report ===")
    by_name = {result.name: result for result in results}
    baseline_off = by_name["baseline_projection_disabled"]
    baseline_on = by_name["baseline_projection_enabled"]
    print(
        "Baseline no-projection final drift: "
        f"dM_occ={baseline_off.drift_stats['occupied']['final']:+.3e}, "
        f"dM_total={baseline_off.drift_stats['total']['final']:+.3e}, "
        f"max_abs dM_occ={baseline_off.drift_stats['occupied']['max_abs']:.3e}, "
        f"slope dM_occ={baseline_off.drift_stats['occupied']['slope']:+.3e}."
    )
    print(
        "Baseline projection-enabled final drift: "
        f"dM_occ={baseline_on.drift_stats['occupied']['final']:+.3e}, "
        f"projection dM_occ={baseline_on.projection_drift['occupied']:+.3e}; "
        f"{projection_consistency_message(baseline_on, 'occupied')}."
    )
    print("Regime scan summary:")
    for result in results:
        print(
            f"  {result.name}: projection={result.projection_enabled}, dt={result.dt:.3e}, "
            f"events={result.projection_events}, final dM_occ={result.drift_stats['occupied']['final']:+.3e}, "
            f"max_abs dM_occ={result.drift_stats['occupied']['max_abs']:.3e}, "
            f"projection dM_occ={result.projection_drift['occupied']:+.3e}, "
            f"slope={result.drift_stats['occupied']['slope']:+.3e}"
        )

    no_projection_roundoff = baseline_off.drift_stats["occupied"]["max_abs"] < args.roundoff_drift_tol
    projection_dominates = (
        baseline_on.projection_events > 0
        and abs(baseline_on.drift_stats["occupied"]["final"] - baseline_on.projection_drift["occupied"])
        <= max(args.projection_match_tol, 0.05 * abs(baseline_on.projection_drift["occupied"]))
    )
    if no_projection_roundoff and projection_dominates:
        conclusion = "Projection is the dominant source of observed long-run occupied-mass drift."
    elif no_projection_roundoff:
        conclusion = "No-projection drift is roundoff-scale; any larger drift is regime/projection dependent."
    else:
        conclusion = "No-projection drift exceeds the configured roundoff tolerance, suggesting slow accumulation or a regime-specific trend."
    print(f"Conclusion: {conclusion}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--dt", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--projection-stress", action="store_true", help="start projection test from an intentionally invalid state")
    parser.add_argument("--long-run", action="store_true", help="run long-run drift stress suite instead of short substep diagnostics")
    parser.add_argument("--lx", type=float, default=50.0, help="long-run domain length in x")
    parser.add_argument("--ly", type=float, default=50.0, help="long-run domain length in y")
    parser.add_argument("--nsteps", type=int, default=None, help="long-run step count; defaults to --steps when provided, otherwise 10000")
    parser.add_argument("--record-every", type=int, default=1000, help="long-run mass/projection reporting interval")
    parser.add_argument("--output-dir", default="diagnostic_outputs/mass_drift_long_run", help="directory for long-run CSV time series; empty disables CSV output")
    parser.add_argument("--include-no-projection-variants", action="store_true", help="also run no-projection controls for the stronger-noise and sharp-interface regimes")
    parser.add_argument("--roundoff-drift-tol", type=float, default=1.0e-9, help="threshold for treating no-projection drift as roundoff-scale")
    parser.add_argument("--projection-match-tol", type=float, default=1.0e-9, help="absolute tolerance for matching final drift to accumulated projection corrections")
    args = parser.parse_args()
    if args.long_run:
        # Long-run defaults follow the requested production-sized diagnostic
        # setup while preserving the smaller historical defaults for short
        # substep accounting runs.
        if args.nx == parser.get_default("nx"):
            args.nx = 128
        if args.ny == parser.get_default("ny"):
            args.ny = 128
        if args.dt == parser.get_default("dt"):
            args.dt = 5.0e-4
    if args.nsteps is None:
        args.nsteps = args.steps if args.long_run and args.steps != parser.get_default("steps") else 10000
    if args.long_run:
        return run_long_run_suite(args)

    lx = ly = 1.0
    rho_A0, rho_B0 = make_random_initial_condition(args.nx, args.ny, rhoA0=0.31, rhoB0=0.29, noise=2e-3, seed=args.seed)

    warnings = 0
    warnings += run_repeated_case(
        "explicit FV only (no projection)",
        build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=False, semi_implicit_stiff=False,
                     simplex_limiter=False, use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0),
        args.steps,
        args.tol,
        ("A", "B", "occupied", "total"),
        explicit_stepper(args.dt),
    ).warnings
    warnings += run_repeated_case(
        "stochastic increment only (no projection)",
        build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=True, semi_implicit_stiff=False,
                     simplex_limiter=False, use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0),
        args.steps,
        args.tol,
        ("occupied", "total"),
        stochastic_stepper(args.dt),
    ).warnings
    warnings += run_repeated_case(
        "semi-implicit Gamma only (no projection)",
        build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=False, semi_implicit_stiff=True,
                     simplex_limiter=False, use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0),
        args.steps,
        args.tol,
        ("A", "B", "occupied", "total"),
        gamma_stepper(args.dt),
    ).warnings

    if args.projection_stress:
        rho_A_proj = rho_A0.copy()
        rho_B_proj = rho_B0.copy()
        rho_A_proj[0, 0] = -0.02
        rho_B_proj[1, 1] = -0.03
        rho_A_proj[2, 2] = 0.8
        rho_B_proj[2, 2] = 0.7
    else:
        rho_A_proj = rho_A0 + 0.0
        rho_B_proj = rho_B0 + 0.0
    run_repeated_case(
        "simplex projection only",
        build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=False, semi_implicit_stiff=False,
                     simplex_limiter=False, use_periodic_fast_path=True, rho_A0=rho_A_proj, rho_B0=rho_B_proj),
        1,
        args.tol,
        tuple(),
        projection_stepper(),
    )

    warnings += compare_fast_and_generic_stochastic(rho_A0, rho_B0, args.nx, args.ny, lx, ly, args.dt, args.seed)

    print("\n=== full combined step accounting (projection disabled) ===")
    full_no_projection = build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=True, semi_implicit_stiff=True,
                                      simplex_limiter=False, use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0)
    for n in range(1, min(args.steps, 5) + 1):
        print(f"\n-- combined no-projection step {n} --")
        warnings += combined_step_accounting(full_no_projection, args.dt, add_noise=True, tol=args.tol)

    print("\n=== full combined step accounting (projection enabled) ===")
    full_projection = build_solver(args.nx, args.ny, lx, ly, args.seed, stochastic=True, semi_implicit_stiff=True,
                                   simplex_limiter=True, use_periodic_fast_path=True, rho_A0=rho_A0, rho_B0=rho_B0)
    for n in range(1, min(args.steps, 5) + 1):
        print(f"\n-- combined projection step {n} --")
        warnings += combined_step_accounting(full_projection, args.dt, add_noise=True, tol=args.tol)

    print("\n=== diagnostic summary ===")
    if warnings:
        print(f"Completed with {warnings} warning(s). Inspect the first warning above to locate the first non-roundoff drift.")
    else:
        print("No non-projection conservation warnings exceeded tolerance.")
    print("Projection-induced mass changes, if any, are printed at projection stages and are expected for local limiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
