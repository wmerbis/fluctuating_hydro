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
from typing import Callable, Dict, Iterable, Tuple

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--dt", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--projection-stress", action="store_true", help="start projection test from an intentionally invalid state")
    args = parser.parse_args()

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
