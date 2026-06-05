#!/usr/bin/env python3
"""Compare limiter modes for long stochastic finite-volume runs.

The report prints final/max drift in species, occupied, and total masses together
with limiter activity counts.  The conservative mode currently targets the fully
active periodic Cartesian explicit/stochastic solver and therefore runs with
``semi_implicit_stiff=False``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_volume_stochastic import ModelParameters, SchellingVoterFVSolver, make_random_initial_condition


def build_params(nx: int, lx: float, voter: float) -> ModelParameters:
    return ModelParameters(
        D_A=0.1,
        D_B=0.12,
        D_v=voter,
        beta=8.0,
        h_noise=lx / nx,
        kappa=np.array([[0.6, -0.4], [-0.3, 0.5]], dtype=float),
        gamma=np.eye(2),
    )


def summarize(name: str, series: Dict[str, np.ndarray], limiter_totals: Dict[str, float]) -> None:
    print(f"\n=== limiter_mode={name} ===")
    for key, label in (("mass_A", "M_A"), ("mass_B", "M_B"), ("mass_occupied", "M_occ"), ("mass_total", "M_total")):
        drift = series[key] - series[key][0]
        print(f"{label:>7}: final drift={drift[-1]:+.6e}  max |drift|={np.max(np.abs(drift)):.6e}")
    print(
        "limiter activity: "
        f"activations={limiter_totals['activations']:.0f}, "
        f"limited_cells={limiter_totals['limited_cells']:.0f}, "
        f"limited_faces={limiter_totals['limited_faces']:.0f}, "
        f"voter_limited_cells={limiter_totals['voter_limited_cells']:.0f}, "
        f"gamma_repair_cells={limiter_totals['gamma_repair_cells']:.0f}, "
        f"max residual violation={limiter_totals['residual_max_violation']:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rhoA0", type=float, default=0.47)
    parser.add_argument("--rhoB0", type=float, default=0.47)
    parser.add_argument("--noise", type=float, default=4.0e-3)
    parser.add_argument("--voter", type=float, default=0.05)
    parser.add_argument("--modes", nargs="+", default=["clip", "conservative"], choices=["none", "clip", "conservative"])
    args = parser.parse_args()

    rho_A0, rho_B0 = make_random_initial_condition(args.nx, args.ny, args.rhoA0, args.rhoB0, args.noise, args.seed)
    params = build_params(args.nx, args.lx, args.voter)

    for offset, mode in enumerate(args.modes):
        solver = SchellingVoterFVSolver(
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            params=params,
            bc="periodic",
            semi_implicit_stiff=False,
            stochastic=True,
            limiter_mode=mode,
            rng=np.random.default_rng(args.seed + 1000 + offset),
            use_periodic_fast_path=True,
        )
        solver.set_state(rho_A0, rho_B0)
        totals = {
            "activations": 0.0,
            "limited_cells": 0.0,
            "limited_faces": 0.0,
            "voter_limited_cells": 0.0,
            "gamma_repair_cells": 0.0,
            "residual_max_violation": 0.0,
        }
        masses = {"mass_A": [], "mass_B": [], "mass_occupied": [], "mass_total": []}
        for step in range(args.steps + 1):
            if step % args.record_every == 0 or step == args.steps:
                m = solver.total_masses()
                masses["mass_A"].append(m["A"])
                masses["mass_B"].append(m["B"])
                masses["mass_occupied"].append(m["occupied"])
                masses["mass_total"].append(m["total"])
            if step == args.steps:
                break
            solver.step(args.dt, add_noise=True)
            stats = solver.last_limiter_stats
            for key in ("activations", "limited_cells", "limited_faces", "voter_limited_cells", "gamma_repair_cells"):
                totals[key] += float(stats.get(key, 0.0))
            totals["residual_max_violation"] = max(totals["residual_max_violation"], float(stats.get("residual_max_violation", 0.0)))
        summarize(mode, {k: np.asarray(v) for k, v in masses.items()}, totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
