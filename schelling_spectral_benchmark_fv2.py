"""Finite-time spectral covariance benchmark for Schelling face reactions.

Drop this script beside your ``fhd/`` package, with the existing module
``schelling_spectrum_fv2.py (beside this benchmark)`` (the fv2 version) installed.

Compares ensembles of *independent stochastic trajectories* with the linear
Gaussian prediction for fluctuations about a homogeneous background:
    S(k,t) = integral_0^t exp[L(k)s] N(k) exp[L(k).T s] ds
plus the propagated initial covariance if --initial-cell-std is nonzero.

Uses the ACTUAL ``fhd.FHD_2D.fhd_2d`` face-reaction sampler and its FE step.
Only periodic boundaries and second-order finite-volume Laplacians are
supported. Theoretical covariance uses the exact Theta response and the fv2
Fourier symbol.  This compares ensemble covariance rather than fitting a
single simulation's noisy radial spectrum.

Example:
  python schelling_spectral_benchmark.py --N 128 128 --L 1 1 \\
      --dt 1e-7 --steps 1 10 50 --replicates 100 \\
      --h 0.0005 0.001 --beta 0 0.63 5 --plot \\
      --output-dir spectral_validation

Start with a small grid and few replicas to smoke-test imports:
  python schelling_spectral_benchmark.py --N 16 16 --dt 1e-6 \\
      --steps 1 3 --replicates 8 --beta 0 --h 0.001

Outputs per beta/h/checkpoint: complete 2x2 S(k,t) in NPZ and radial CSV
(and optional figures). A summary.csv contains integrated variance ratios.

Notes:
  * For a homogeneous INITIAL state, initial_cell_std=(0,0) is exact.
  * h sets cell capacity Omega=dx*dy/h**2, density quantum 1/Omega.
    If the quantum exceeds donor/receiver stocks, events can be blocked.
  * Mode-by-mode agreement can be noisy at finite ensemble size. Comparisons
    of ring means and integrated spectral variance are more statistically
    stable. Bin SEM treats independent REPLICAS as its sampling units.
  * A finite-time linear theory is not quantitatively expected once nonlinear
    segregation or strong depletion develops.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

try:
    from fhd.FHD_2D import fhd_2d
except ImportError as exc:
    raise ImportError(
        "Run from the repository root so fhd/FHD_2D.py and fhd/operations.py can be imported."
    ) from exc

# First look alongside this script; alternatively accept the module in fhd/.
try:
    from schelling_spectrum_fv2 import (
        spectrum, spatial_symbols, check_periodic_solver_laplacian,
    )
except ModuleNotFoundError as exc:
    if exc.name != "schelling_spectrum_fv2":
        raise
    try:
        from fhd.schelling_spectrum_fv2 import (
            spectrum, spatial_symbols, check_periodic_solver_laplacian,
        )
    except ModuleNotFoundError as inner:
        if inner.name not in ("fhd.schelling_spectrum_fv2", "schelling_spectrum_fv2"):
            raise
        raise ModuleNotFoundError(
            "Missing schelling_spectrum_fv2.py. Put it beside this benchmark "
            "in the repository root or inside the fhd/ package."
        ) from inner


# A and B are species. +/- are normalized total/segregation eigenmode fields.
COMPONENTS = ("AA", "AB", "BB", "plus", "minus")


def components(S):
    """Return named (Nx,Ny) spectra from a 2x2 matrix spectrum."""
    aa, ab, bb = S[0, 0].real, S[0, 1].real, S[1, 1].real
    return {
        "AA": aa,
        "AB": ab,
        "BB": bb,
        "plus": 0.5 * (aa + bb + 2 * ab),
        "minus": 0.5 * (aa + bb - 2 * ab),
    }


class RadialBinner:
    """Azimuthal bins based on physical |k|; k=0 excluded throughout."""
    def __init__(self, N, L, nbins):
        kx, ky, _, _ = spatial_symbols(N, L, symbol="fv2")
        kr = np.hypot(kx, ky)
        edges = np.linspace(0.0, kr.max() * (1 + 1e-12), nbins + 1)
        ids = np.searchsorted(edges, kr.ravel(), side="right") - 1
        nonzero = kr.ravel() > 0
        self.ids = ids[nonzero]
        self.mask = nonzero
        self.nbins = nbins
        self.k = 0.5 * (edges[:-1] + edges[1:])
        self.count = np.bincount(self.ids, minlength=nbins)

    def radial(self, image):
        vals = np.asarray(image).ravel()[self.mask]
        finite = np.isfinite(vals)
        sums = np.bincount(self.ids[finite], weights=vals[finite], minlength=self.nbins)
        n = np.bincount(self.ids[finite], minlength=self.nbins)
        return np.divide(sums, n, where=n != 0,
                         out=np.full(self.nbins, np.nan))

    def all_components(self, S):
        return {name: self.radial(values) for name, values in components(S).items()}


class SpectralMoments:
    """Online unbiased cross-covariance of Fourier amplitudes across runs."""
    def __init__(self, N, L, binner):
        self.N = tuple(N)
        self.L = tuple(L)
        self.binner = binner
        self.area = float(np.prod(L))
        self.cells = int(np.prod(N))
        self.scale = np.sqrt(self.area) / self.cells
        self.count = 0
        self.sum_fft = np.zeros((2, *N), dtype=np.complex128)
        self.sum_outer = np.zeros((2, 2, *N), dtype=np.complex128)
        self.bin_sum = {key: np.zeros(binner.nbins) for key in COMPONENTS}
        self.bin_sum2 = {key: np.zeros(binner.nbins) for key in COMPONENTS}

    def add(self, phi):
        # The k=0 mode vanishes identically; each realization conserves mass.
        centered = phi - phi.mean(axis=(1, 2), keepdims=True)
        F = self.scale * np.fft.fft2(centered, axes=(-2, -1))
        outer = np.einsum("axy,bxy->abxy", F, F.conj(), optimize=True)
        self.count += 1
        self.sum_fft += F
        self.sum_outer += outer
        # SEM of shell averages across independent runs, NOT across k modes.
        for key, binned in self.binner.all_components(outer).items():
            good = np.isfinite(binned)
            self.bin_sum[key][good] += binned[good]
            self.bin_sum2[key][good] += binned[good]**2

    def estimate(self):
        R = self.count
        if R < 2:
            raise ValueError("Need at least two independent stochastic trajectories")
        mean_outer = np.einsum("axy,bxy->abxy", self.sum_fft,
                               self.sum_fft.conj(), optimize=True) / R
        # Unbiased estimate of covariance across trajectories, mode by mode.
        S = (self.sum_outer - mean_outer) / (R - 1)
        S = 0.5 * (S + np.conjugate(S.swapaxes(0, 1)))
        S[:, :, 0, 0] = 0.0
        shell = self.binner.all_components(S.real)
        sem = {}
        for key in COMPONENTS:
            raw_mean = self.bin_sum[key] / R
            sample_var = (self.bin_sum2[key] - R * raw_mean**2) / (R - 1)
            sem[key] = np.sqrt(np.maximum(0, sample_var) / R)
            sem[key][self.binner.count == 0] = np.nan
        return S, shell, sem


def initial_field(N, rho, initial_std, rng):
    phi = np.broadcast_to(np.asarray(rho, dtype=float)[:, None, None], (2, *N)).copy()
    if np.any(initial_std):
        # Random initial Fourier modes, but preserve the mean species masses.
        perturb = rng.normal(size=phi.shape) * np.asarray(initial_std)[:, None, None]
        perturb -= perturb.mean(axis=(1, 2), keepdims=True)
        phi += perturb
        if np.min(phi) < 0.0 or np.max(phi.sum(axis=0)) > 1.0:
            raise ValueError("Initial density left simplex: reduce --initial-cell-std")
    return phi


def rel_l2(emp, th, mask):
    x, y = emp[mask], th[mask]
    den = np.linalg.norm(y)
    return float(np.linalg.norm(x - y) / den) if den else np.nan


def save_radial(path, binner, observed, expected, sem):
    fields = ["k", "modes"]
    for label in COMPONENTS:
        fields += [f"emp_{label}", f"theory_{label}", f"sem_{label}"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(binner.nbins):
            if binner.count[i] == 0:
                continue
            row = dict(k=float(binner.k[i]), modes=int(binner.count[i]))
            for label in COMPONENTS:
                row[f"emp_{label}"] = float(observed[label][i])
                row[f"theory_{label}"] = float(expected[label][i])
                row[f"sem_{label}"] = float(sem[label][i])
            writer.writerow(row)


def plot_radial(out_prefix, binner, observed, expected, sem):
    import matplotlib.pyplot as plt
    for label in COMPONENTS:
        good = (binner.count > 0) & np.isfinite(observed[label]) & np.isfinite(expected[label])
        x = binner.k[good]
        o, t, err = observed[label][good], expected[label][good], sem[label][good]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, t, lw=2, label="linear FHD theory")
        ax.plot(x, o, lw=1.4, label="face-reaction ensemble")
        ax.fill_between(x, o - 2*err, o + 2*err, alpha=0.15,
                        label="ensemble ±2 SEM")
        if label != "AB" and np.all(t > 0) and np.all(o > 0):
            ax.set_yscale("log")
        ax.set_xlabel(r"$|\mathbf{k}|$")
        ax.set_ylabel(r"$S_{" + label + r"}(k,t)$")
        ax.set_title(f"Finite-time spectral covariance: {label}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{label}.png", dpi=170)
        plt.close(fig)


def benchmark_case(solver, param, args, binner, rng, case_prefix):
    omega = solver.cell_area / param["h"]**2
    eps = 1 / omega
    if omega < 20:
        print(f"WARNING: Omega={omega:.2f} is small; the density quantum "
              f"epsilon={eps:.5g} may make discrete hopping strongly non-Gaussian")
    if eps >= min(args.rho[0], args.rho[1], 1 - sum(args.rho)):
        print(f"WARNING: epsilon={eps:.5g} exceeds a donor/vacancy fraction; "
              "some or all hopping channels will be blocked")
    work = solver._ensure_work()
    # Caches depend on beta and h. Prepare separately for each case.
    solver.prepare_schelling(initial_field(tuple(args.N), args.rho,
                                          (0.0, 0.0), rng), param)
    moments = {n: SpectralMoments(tuple(args.N), tuple(args.L), binner)
               for n in args.steps}
    mass_error = 0.0
    min_density, min_vacancy = 1.0, 1.0
    tic = time.perf_counter()
    for replica in range(args.replicates):
        solver.set_seed(args.seed + replica + args.case_index * 1_000_003)
        phi = initial_field(tuple(args.N), args.rho, args.initial_cell_std, rng)
        initial_mass = phi.sum(axis=(1, 2))
        for step in range(1, max(args.steps) + 1):
            phi = solver.step(solver.rhs_Schelling, phi, param,
                              args.dt, True, "FE", work=work,
                              step=step, record_projection_history=False)
            if step in moments:
                # Add Fourier coefficients immediately: solver's work buffer
                # will be overwritten by the next step / next replica.
                moments[step].add(phi)
        mass_error = max(mass_error, float(np.max(np.abs(phi.sum(axis=(1,2))-initial_mass))))
        min_density = min(min_density, float(np.min(phi)))
        min_vacancy = min(min_vacancy, float(np.min(1.0 - phi.sum(axis=0))))
        if (replica + 1) % max(1, args.replicates // 5) == 0:
            print(f"  {replica+1}/{args.replicates} replicas, elapsed {time.perf_counter()-tic:.1f}s", flush=True)

    rows = []
    for n in args.steps:
        t = n * args.dt
        print(f"  Comparing at step {n}, physical time {t:.6g} ...", flush=True)
        theory = spectrum(N=args.N, L=args.L, rho=args.rho, D=args.D,
                          beta=param["beta"], kappa=param["kappa"],
                          h=param["h"], D_v=0., symbol="fv2", time=t,
                          initial_cell_std=args.initial_cell_std)
        emp_S, emp_shell, emp_sem = moments[n].estimate()
        theory_S = theory["S"]
        if float(theory["growth"][..., -1].max()) * t > 1.0:
            print("    WARNING: fastest linear instability has grown by "
                  "more than e-fold; nonlinear effects may invalidate this comparison")
        theory_shell = binner.all_components(theory_S)
        mask = theory["q2"] > 0
        row = {
            "beta": float(param["beta"]), "h": float(param["h"]),
            "dt": args.dt, "step": n, "time": t,
            "N": f"{args.N[0]}x{args.N[1]}", "replicates": args.replicates,
            "omega": omega, "epsilon": eps,
            "max_mass_drift": mass_error,
            "min_species": min_density, "min_vacancy": min_vacancy,
            "max_linear_growth_rate": float(theory["growth"][..., -1].max()),
        }
        ecomp, tcomp = components(emp_S), components(theory_S)
        for label in COMPONENTS:
            theory_total = float(tcomp[label][mask].sum())
            empirical_total = float(ecomp[label][mask].sum())
            row[f"integrated_ratio_{label}"] = (empirical_total / theory_total
                                                   if abs(theory_total) > 1e-30 else np.nan)
            row[f"full_k_relative_L2_{label}"] = rel_l2(ecomp[label], tcomp[label], mask)
            filled = (binner.count > 0) & np.isfinite(theory_shell[label])
            row[f"radial_relative_L2_{label}"] = rel_l2(emp_shell[label], theory_shell[label], filled)
        # Cross-covariance may be near zero (e.g., at very early times),
        # making relative errors w.r.t. AB itself misleadingly enormous.
        good = (binner.count > 0) & np.isfinite(theory_shell["AB"])
        cross_scale = np.linalg.norm(0.5 * (theory_shell["AA"][good] + theory_shell["BB"][good]))
        row["radial_AB_error_over_auto_power"] = (float(np.linalg.norm(
            emp_shell["AB"][good] - theory_shell["AB"][good]) / cross_scale)
            if cross_scale > 0 else np.nan)
        name = f"{case_prefix}_step{n:05d}"
        np.savez_compressed(args.output_dir / f"{name}.npz",
                            S_emp=emp_S, S_theory=theory_S,
                            q2=theory["q2"], kx=theory["kx"], ky=theory["ky"],
                            growth=theory["growth"],
                            beta=param["beta"], h=param["h"], dt=args.dt,
                            time=t, replicates=args.replicates,
                            initial_cell_std=args.initial_cell_std)
        save_radial(args.output_dir / f"{name}_radial.csv", binner,
                    emp_shell, theory_shell, emp_sem)
        if args.plot:
            plot_radial(args.output_dir / name, binner,
                        emp_shell, theory_shell, emp_sem)
        rows.append(row)
        print("    Integrated variance ratios: AA={:.3f}, BB={:.3f}, += {:.3f}, -= {:.3f}".format(
              row["integrated_ratio_AA"], row["integrated_ratio_BB"],
              row["integrated_ratio_plus"], row["integrated_ratio_minus"]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", nargs=2, type=int, default=(128, 128))
    parser.add_argument("--L", nargs=2, type=float, default=(1.0, 1.0))
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, nargs="+", default=(1, 5, 20),
                        help="Sampling times in INTEGER steps; physical times are steps*dt")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--rho", type=float, nargs=2, default=(.35, .35))
    parser.add_argument("--D", type=float, nargs=2, default=(1., 1.))
    parser.add_argument("--kappa", type=float, nargs=4, default=(.6, -.4, -.4, .6))
    parser.add_argument("--beta", nargs="+", type=float, default=(0., .63, 5.))
    parser.add_argument("--h", nargs="+", type=float, default=(.0005, .001))
    parser.add_argument("--theta-face", choices=("midpoint", "endpoints", "gauss3"),
                        default="midpoint")
    parser.add_argument("--initial-cell-std", type=float, nargs=2, default=(0., 0.),
                        help="Optional independent per-cell initial Gaussian fluctuations")
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--seed", type=int, default=24051)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("schelling_spectral_results"))
    args = parser.parse_args()
    if args.dt <= 0 or args.replicates < 2 or args.bins < 4:
        parser.error("dt must be positive, replicates >= 2, bins >= 4")
    if min(args.N) < 4 or min(args.L) <= 0:
        parser.error("N >= 4 and domain lengths positive are required")
    if min(args.rho) <= 0 or sum(args.rho) >= 1:
        parser.error("require rho_A,rho_B>0 and rho_A+rho_B<1")
    if np.any(np.asarray(args.initial_cell_std) < 0):
        parser.error("initial standard deviations must be nonnegative")
    if any(step <= 0 for step in args.steps):
        parser.error("sampling steps must be positive")
    args.steps = sorted(set(args.steps))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    solver = fhd_2d(L=tuple(args.L), N=tuple(args.N), bc="periodic", fft=False,
                    schelling_flux="face_reaction", projection_floor=0.0,
                    projection_mode="redistribute", reaction_diagnostic=False)
    errors = check_periodic_solver_laplacian(solver)
    worst_error = max(errors.values())
    print("Second-order solver Laplacian check:", errors, flush=True)
    if worst_error > 1e-6:
        raise RuntimeError(
            "The imported solver does not use the expected second-order periodic Laplacian. "
            "Make sure your fhd/FHD_2D.py uses makeD2_second_order for D2x and D2y. "
            "The theory uses symbol='fv2', so mismatched operators invalidate the test."
        )
    binner = RadialBinner(tuple(args.N), tuple(args.L), args.bins)
    rng = np.random.default_rng(args.seed)
    rows = []
    tick = time.perf_counter()
    args.case_index = 0
    for beta in args.beta:
        for h in args.h:
            param = {
                "D": np.asarray(args.D), "beta": float(beta),
                "kappa": np.asarray(args.kappa).reshape(2, 2),
                "rho_bar": np.asarray(args.rho), "h": float(h),
                "theta_face": args.theta_face,
            }
            prefix = f"beta{beta:g}_h{h:g}".replace("-", "m")
            print(f"Case beta={beta:g}, h={h:g}, dt={args.dt:g}, "
                  f"steps={args.steps}, replicas={args.replicates}", flush=True)
            rows.extend(benchmark_case(solver, param, args, binner, rng, prefix))
            args.case_index += 1
    with (args.output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Done:", args.output_dir.resolve(),
          f"({time.perf_counter()-tick:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
