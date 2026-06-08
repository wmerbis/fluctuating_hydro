#!/usr/bin/env python3
"""Compare ``none`` and ``clip`` limiter dynamics for matched FV runs.

The script uses identical initial conditions and RNG seeds for each limiter mode,
then records mass drift, segregation diagnostics, Fourier peak strength, and the
clip projection activity that remains in the simplified solver.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fhd
from finite_volume_stochastic import ModelParameters, SchellingVoterFVSolver, make_random_initial_condition


def build_params(nx: int, lx: float, voter: float, beta: float) -> ModelParameters:
    return ModelParameters(
        D_A=0.1,
        D_B=0.12,
        D_v=voter,
        beta=beta,
        h_noise=lx / nx,
        kappa=np.array([[0.6, -0.4], [-0.3, 0.5]], dtype=float),
        gamma=np.eye(2),
    )


def structure_metrics(rho_A: np.ndarray, rho_B: np.ndarray) -> Dict[str, float]:
    order = rho_A - rho_B
    centered = order - float(np.mean(order))
    spectrum = np.abs(np.fft.rfft2(centered))
    spectrum[0, 0] = 0.0
    peak_index = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    phi = np.array([rho_A.T, rho_B.T])
    return {
        "dissimilarity": float(fhd.dissimilarity(phi)),
        "mean_relative_entropy": float(fhd.mean_relative_entropy(phi)),
        "order_variance": float(np.var(order)),
        "order_rms": float(np.sqrt(np.mean(centered * centered))),
        "fourier_peak_amplitude": float(spectrum[peak_index]),
        "fourier_peak_ky_index": float(peak_index[0]),
        "fourier_peak_kx_index": float(peak_index[1]),
    }


def summarize(name: str, series: Dict[str, np.ndarray], limiter_totals: Dict[str, float]) -> None:
    print(f"\n=== limiter_mode={name} ===")
    for key, label in (("mass_A", "M_A"), ("mass_B", "M_B"), ("mass_occupied", "M_occ"), ("mass_total", "M_total")):
        drift = series[key] - series[key][0]
        print(f"{label:>7}: final drift={drift[-1]:+.6e}  max |drift|={np.max(np.abs(drift)):.6e}")
    for key in ("dissimilarity", "order_variance", "fourier_peak_amplitude"):
        print(f"{key:>25}: initial={series[key][0]:.6e} final={series[key][-1]:.6e} max={np.max(series[key]):.6e}")
    print(
        "limiter activity: "
        f"activations={limiter_totals['activations']:.0f}, "
        f"limited_cells={limiter_totals['limited_cells']:.0f}, "
        f"max residual violation={limiter_totals['residual_max_violation']:.3e}, "
        f"projection ΔM_occ={limiter_totals['projection_delta_occupied']:+.6e}"
    )


def run_one(args: argparse.Namespace, mode: str, params: ModelParameters, rho_A0: np.ndarray, rho_B0: np.ndarray) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
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
        rng=np.random.default_rng(args.seed + 1000),
        use_periodic_fast_path=True,
    )
    solver.set_state(rho_A0, rho_B0)
    totals = {
        "activations": 0.0,
        "limited_cells": 0.0,
        "residual_max_violation": 0.0,
        "projection_delta_occupied": 0.0,
    }
    series: Dict[str, list[float]] = {"time": [], "mass_A": [], "mass_B": [], "mass_occupied": [], "mass_total": []}
    for key in structure_metrics(solver.rho_A, solver.rho_B):
        series[key] = []

    for step in range(args.steps + 1):
        if step % args.record_every == 0 or step == args.steps:
            m = solver.total_masses()
            series["time"].append(step * args.dt)
            for key, mass_key in (("mass_A", "A"), ("mass_B", "B"), ("mass_occupied", "occupied"), ("mass_total", "total")):
                series[key].append(m[mass_key])
            for key, value in structure_metrics(solver.rho_A, solver.rho_B).items():
                series[key].append(value)
        if step == args.steps:
            break
        solver.step(args.dt, add_noise=True)
        stats = solver.last_limiter_stats
        totals["activations"] += float(stats.get("activations", 0.0))
        totals["limited_cells"] += float(stats.get("limited_cells", 0.0))
        totals["projection_delta_occupied"] += float(stats.get("projection_delta_occupied", 0.0))
        totals["residual_max_violation"] = max(totals["residual_max_violation"], float(stats.get("residual_max_violation", 0.0)))
    return {k: np.asarray(v) for k, v in series.items()}, totals


def parse_scan(values: str) -> Iterable[float]:
    for item in values.split(","):
        item = item.strip()
        if item:
            yield float(item)


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
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--beta-scan", type=str, default="", help="Comma-separated beta values for a small instability-threshold scan.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for final mode/scan summary rows.")
    parser.add_argument("--modes", nargs="+", default=["none", "clip"], choices=["none", "clip"])
    args = parser.parse_args()

    scan_betas = list(parse_scan(args.beta_scan)) if args.beta_scan else [args.beta]
    rows = []
    for beta in scan_betas:
        print(f"\n######## beta={beta:g} ########")
        rho_A0, rho_B0 = make_random_initial_condition(args.nx, args.ny, args.rhoA0, args.rhoB0, args.noise, args.seed)
        params = build_params(args.nx, args.lx, args.voter, beta)
        for mode in args.modes:
            series, totals = run_one(args, mode, params, rho_A0, rho_B0)
            summarize(mode, series, totals)
            rows.append(
                {
                    "beta": beta,
                    "mode": mode,
                    "final_dissimilarity": series["dissimilarity"][-1],
                    "final_order_variance": series["order_variance"][-1],
                    "max_order_variance": np.max(series["order_variance"]),
                    "final_fourier_peak_amplitude": series["fourier_peak_amplitude"][-1],
                    "occupied_mass_drift": series["mass_occupied"][-1] - series["mass_occupied"][0],
                    "limited_cells": totals["limited_cells"],
                    "projection_delta_occupied": totals["projection_delta_occupied"],
                }
            )
    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
