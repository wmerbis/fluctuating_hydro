"""
benchmark_schelling_face_sampler.py

Benchmark the old Gaussian finite-volume Schelling face update against
the new exact directed-reaction / pure-death face sampler.

Checks
------
1. Mean one-step increment.
2. Variance of one-step increment.
3. Agreement with small-dt Kramers-Moyal predictions.
4. Species-A mass conservation across the face.
5. Number/rate of simplex violations.
6. Boundary stress tests.

IMPORTANT
---------
The NEW sampler represents passive Schelling diffusion + its fluctuations.
Therefore the OLD comparison below also includes the deterministic passive
FV drift in addition to the Gaussian conservative noise.

Adjust the import line below if your package path differs.
"""

import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Adjust this import if needed
# ---------------------------------------------------------------------
from fhd.FHD_2D import fhd_2d

# =====================================================================
# USER SETTINGS
# =====================================================================

SEED = 12345

# Number of Monte Carlo samples per state / dt.
# Start with 20_000; increase to 100_000 for final statistics.
N_TRIALS = 100_000

# Grid geometry.  Taking dx=dy=1 makes interpretation easy.
DX = 1.0
DY = 1.0
CELL_VOLUME = DX * DY

# Passive Schelling diffusion coefficient for species A.
D_A = 0.1

# Effective microscopic number of sites per hydrodynamic cell.
OMEGA = 10_000

# h^2 = V / Omega
H = np.sqrt(CELL_VOLUME / OMEGA)

# Production-like small timesteps.
DT_VALUES = [0.002, 0.005, 0.02, 0.05]

# Optional deliberately large dt, used ONLY for boundary states,
# so violations from the Gaussian scheme become visible.
STRESS_DT = 2.0

# Species being tested.
SPECIES = 0  # A

TOL = 1e-13


# =====================================================================
# MODEL INSTANCE
# =====================================================================

# Use a grid large enough for the existing finite-difference operators.
# Only two cells are actually touched by the reaction benchmark.
sim = fhd_2d(
    L=(8.0, 8.0),
    N=(8, 8),
    bc="periodic",
    fft=False,
    schelling_flux="finite_volume",
)

sim.set_seed(SEED)

LEFT = (3, 3)
RIGHT = (4, 3)


# =====================================================================
# INITIAL STATES
# =====================================================================

EPS = 1.0 / OMEGA

# Entries are
#   (A_L, B_L, A_R, B_R)
#
# Homogeneous cases are especially useful because the old arithmetic
# face mobility and the new cross-reaction mobility have exactly the
# same leading-order variance there.

CASES = {
    "interior_homogeneous": (
        0.35, 0.35,
        0.35, 0.35,
    ),

    "smooth_gradient": (
        0.34, 0.35,
        0.36, 0.35,
    ),

    # Exactly one microscopic A quantum in each cell.
    "one_quantum_A": (
        EPS, 0.35,
        EPS, 0.35,
    ),

    # Exactly one vacancy quantum in each cell.
    "one_quantum_vacancy": (
        1.0 - 0.35 - EPS, 0.35,
        1.0 - 0.35 - EPS, 0.35,
    ),
}


# =====================================================================
# BASIC UTILITIES
# =====================================================================

def vacancies(aL, bL, aR, bR):
    vL = 1.0 - aL - bL
    vR = 1.0 - aR - bR
    return vL, vR


def assert_valid_initial_state(aL, bL, aR, bR):
    vL, vR = vacancies(aL, bL, aR, bR)

    vals = (aL, bL, vL, aR, bR, vR)

    if min(vals) < -TOL:
        raise ValueError(
            "Invalid initial condition: "
            f"A_L={aL}, B_L={bL}, V_L={vL}, "
            f"A_R={aR}, B_R={bR}, V_R={vR}"
        )


def simplex_violation_mask(aL, aR, bL, bR, tol=TOL):
    """
    Return Boolean mask for trials in which either cell violates

        A >= 0
        B >= 0
        A + B <= 1.

    B is fixed in this benchmark.
    """
    vL = 1.0 - aL - bL
    vR = 1.0 - aR - bR

    bad_L = (
        (aL < -tol)
        | (bL < -tol)
        | (vL < -tol)
        | (aL > 1.0 + tol)
        | (bL > 1.0 + tol)
    )

    bad_R = (
        (aR < -tol)
        | (bR < -tol)
        | (vR < -tol)
        | (aR > 1.0 + tol)
        | (bR > 1.0 + tol)
    )

    return bad_L | bad_R


# =====================================================================
# THEORETICAL SMALL-dt MOMENTS
# =====================================================================

def theoretical_moments(aL, bL, aR, bR, D, delta, dt, h):
    """
    Small-dt first two moments for a single face.

    Returns
    -------
    mean_increment
        Common O(dt) drift for both schemes.

    var_old
        Variance used by the current Gaussian FV discretization:
            M_f = 0.5 * (A_L V_L + A_R V_R).

    var_reaction
        Variance from the exact cross-face chemical reactions:
            A_L V_R + A_R V_L.

    These agree exactly for homogeneous L/R states and differ only
    at second spatial order for smooth fields.
    """
    vL, vR = vacancies(aL, bL, aR, bR)

    # Centered passive finite-volume drift.
    drift = (
        D / delta**2
        * (vL * aR - aL * vR)
    )

    mean_increment = drift * dt

    # --------------------------------------------------------------
    # Current FV Gaussian mobility:
    #
    # M_f = 1/2 [ A_L V_L + A_R V_R ]
    #
    # Delta A_L noise =
    #     h / delta *
    #     sqrt(2 D M_f dt / V) Z
    # --------------------------------------------------------------
    M_old = 0.5 * (
        aL * vL
        + aR * vR
    )

    var_old = (
        2.0
        * D
        * h**2
        * M_old
        * dt
        / (CELL_VOLUME * delta**2)
    )

    # --------------------------------------------------------------
    # Directed-reaction Kramers-Moyal variance:
    #
    # D h^2/(V delta^2)
    # [A_L V_R + A_R V_L] dt
    # --------------------------------------------------------------
    var_reaction = (
        D
        * h**2
        * (
            aL * vR
            + aR * vL
        )
        * dt
        / (CELL_VOLUME * delta**2)
    )

    return mean_increment, var_old, var_reaction


# =====================================================================
# OLD GAUSSIAN FV FORWARD-EULER UPDATE
# =====================================================================

def sample_old_gaussian(
    rng,
    aL,
    bL,
    aR,
    bR,
    D,
    delta,
    dt,
    h,
    n_trials,
):
    """
    Single-face version of the current finite-volume Gaussian method.

    Includes:
        1. passive centered FV drift via forward Euler;
        2. current arithmetic-mobility Gaussian face noise.

    The update is conservative:
        delta A_R = -delta A_L.
    """
    vL, vR = vacancies(aL, bL, aR, bR)

    # Passive finite-volume drift.
    drift_increment = (
        D / delta**2
        * (vL * aR - aL * vR)
        * dt
    )

    # Current arithmetic face mobility.
    M_face = 0.5 * (
        aL * vL
        + aR * vR
    )

    noise_std = (
        h
        / delta
        * np.sqrt(
            2.0
            * D
            * M_face
            * dt
            / CELL_VOLUME
        )
    )

    z = rng.standard_normal(n_trials)

    delta_aL = drift_increment + noise_std * z

    out_aL = aL + delta_aL
    out_aR = aR - delta_aL

    return out_aL, out_aR


# =====================================================================
# NEW PURE-DEATH REACTION UPDATE
# =====================================================================

def sample_new_reaction(
    sim,
    aL,
    bL,
    aR,
    bR,
    D,
    delta,
    dt,
    h,
    n_trials,
):
    """
    Use the new package implementation directly.

    Assumes you added the previously discussed method

        sim._passive_face_reaction_strang(...)

    with signature:

        _passive_face_reaction_strang(
            phi,
            species,
            left,
            right,
            D_a,
            delta,
            dt,
            h,
            reverse_order=False,
        )

    If you changed the method name/signature, this is the only function
    in the benchmark you should need to edit.
    """

    out_aL = np.empty(n_trials)
    out_aR = np.empty(n_trials)

    # Background field is irrelevant because only LEFT and RIGHT
    # are touched, but keep it safely inside the simplex.
    phi_template = np.empty((2, 8, 8), dtype=float)
    phi_template[0, :, :] = 0.30
    phi_template[1, :, :] = 0.30

    for r in range(n_trials):

        phi = phi_template.copy()

        phi[0, LEFT[0], LEFT[1]] = aL
        phi[1, LEFT[0], LEFT[1]] = bL

        phi[0, RIGHT[0], RIGHT[1]] = aR
        phi[1, RIGHT[0], RIGHT[1]] = bR

        # Alternate ABA / BAB orientation between Monte Carlo samples.
        # This removes any systematic left/right preference in the benchmark.
        reverse = bool(r & 1)

        sim._passive_face_reaction_strang(
            phi=phi,
            species=SPECIES,
            left=LEFT,
            right=RIGHT,
            D_a=D,
            delta=delta,
            dt=dt,
            h=h,
            reverse_order=reverse,
        )

        out_aL[r] = phi[SPECIES, LEFT[0], LEFT[1]]
        out_aR[r] = phi[SPECIES, RIGHT[0], RIGHT[1]]

    return out_aL, out_aR


# =====================================================================
# STATISTICS
# =====================================================================

def summarize_samples(
    name,
    aL0,
    aR0,
    bL,
    bR,
    out_aL,
    out_aR,
    theory_mean,
    theory_var,
):
    dA = out_aL - aL0

    mean_emp = float(np.mean(dA))
    var_emp = float(np.var(dA, ddof=1))

    if theory_var > 0.0:
        variance_ratio = var_emp / theory_var
        variance_rel_error = (var_emp - theory_var) / theory_var
    else:
        variance_ratio = np.nan
        variance_rel_error = np.nan

    # Monte Carlo standard error of the mean.
    sem = np.sqrt(var_emp / len(dA))

    if sem > 0:
        mean_zscore = (mean_emp - theory_mean) / sem
    else:
        mean_zscore = np.nan

    bad = simplex_violation_mask(
        out_aL,
        out_aR,
        bL,
        bR,
    )

    n_bad = int(np.count_nonzero(bad))
    violation_rate = n_bad / len(dA)

    # Species conservation across the face.
    mass_error = (
        out_aL + out_aR
        - (aL0 + aR0)
    )

    max_mass_error = float(
        np.max(np.abs(mass_error))
    )

    return {
        "method": name,
        "mean_emp": mean_emp,
        "mean_theory": theory_mean,
        "mean_zscore": mean_zscore,
        "var_emp": var_emp,
        "var_theory": theory_var,
        "var_ratio": variance_ratio,
        "var_rel_error": variance_rel_error,
        "n_violations": n_bad,
        "violation_rate": violation_rate,
        "max_mass_error": max_mass_error,
    }


def print_summary(case_name, dt, old, new, old_var_th, new_var_th):
    print()
    print("=" * 100)
    print(f"CASE: {case_name:25s}     dt = {dt:g}")
    print("=" * 100)

    print(
        f"{'method':14s} "
        f"{'mean(emp)':>13s} "
        f"{'mean(th)':>13s} "
        f"{'mean z':>10s} "
        f"{'var(emp)':>13s} "
        f"{'var(th)':>13s} "
        f"{'var/th':>10s} "
        f"{'violations':>12s} "
        f"{'viol.rate':>11s}"
    )

    for result in (old, new):
        print(
            f"{result['method']:14s} "
            f"{result['mean_emp']:+13.5e} "
            f"{result['mean_theory']:+13.5e} "
            f"{result['mean_zscore']:+10.3f} "
            f"{result['var_emp']:13.5e} "
            f"{result['var_theory']:13.5e} "
            f"{result['var_ratio']:10.5f} "
            f"{result['n_violations']:12d} "
            f"{result['violation_rate']:11.5e}"
        )

    print()
    print(
        "Leading-order variance targets:  "
        f"old FV Gaussian = {old_var_th:.6e},  "
        f"reaction process = {new_var_th:.6e},  "
        f"ratio reaction/old = {new_var_th / old_var_th:.8f}"
        if old_var_th > 0
        else ""
    )

    print(
        "max |A_L + A_R mass error|: "
        f"old={old['max_mass_error']:.3e}, "
        f"new={new['max_mass_error']:.3e}"
    )


# =====================================================================
# RUN ONE CASE
# =====================================================================

def run_case(case_name, state, dt, seed):
    aL, bL, aR, bR = state

    assert_valid_initial_state(
        aL, bL, aR, bR
    )

    vL, vR = vacancies(
        aL, bL, aR, bR
    )

    print()
    print(
        f"Initial state: "
        f"L=(A={aL:.8f}, B={bL:.8f}, V={vL:.8f}), "
        f"R=(A={aR:.8f}, B={bR:.8f}, V={vR:.8f})"
    )

    theory_mean, var_old_th, var_new_th = theoretical_moments(
        aL=aL,
        bL=bL,
        aR=aR,
        bR=bR,
        D=D_A,
        delta=DX,
        dt=dt,
        h=H,
    )

    # --------------------------------------------------------------
    # Old Gaussian scheme
    # --------------------------------------------------------------
    rng_old = np.random.default_rng(seed)

    t0 = time.perf_counter()

    old_aL, old_aR = sample_old_gaussian(
        rng=rng_old,
        aL=aL,
        bL=bL,
        aR=aR,
        bR=bR,
        D=D_A,
        delta=DX,
        dt=dt,
        h=H,
        n_trials=N_TRIALS,
    )

    old_seconds = time.perf_counter() - t0

    # --------------------------------------------------------------
    # New reaction scheme
    # --------------------------------------------------------------
    sim.set_seed(seed + 1)

    t0 = time.perf_counter()

    new_aL, new_aR = sample_new_reaction(
        sim=sim,
        aL=aL,
        bL=bL,
        aR=aR,
        bR=bR,
        D=D_A,
        delta=DX,
        dt=dt,
        h=H,
        n_trials=N_TRIALS,
    )

    new_seconds = time.perf_counter() - t0

    old_summary = summarize_samples(
        name="Gaussian-FE",
        aL0=aL,
        aR0=aR,
        bL=bL,
        bR=bR,
        out_aL=old_aL,
        out_aR=old_aR,
        theory_mean=theory_mean,
        theory_var=var_old_th,
    )

    new_summary = summarize_samples(
        name="Pure-death",
        aL0=aL,
        aR0=aR,
        bL=bL,
        bR=bR,
        out_aL=new_aL,
        out_aR=new_aR,
        theory_mean=theory_mean,
        theory_var=var_new_th,
    )

    print_summary(
        case_name,
        dt,
        old_summary,
        new_summary,
        var_old_th,
        var_new_th,
    )

    print(
        f"runtime: Gaussian={old_seconds:.3f}s, "
        f"pure-death={new_seconds:.3f}s"
    )

    return {
        "case": case_name,
        "dt": dt,
        "old": old_summary,
        "new": new_summary,
        "old_aL": old_aL,
        "new_aL": new_aL,
        "aL0": aL,
    }


# =====================================================================
# MAIN BENCHMARK
# =====================================================================

def main():

    print("=" * 100)
    print("PURE-DEATH vs GAUSSIAN CONSERVATIVE FACE BENCHMARK")
    print("=" * 100)

    print(f"N_TRIALS = {N_TRIALS}")
    print(f"D_A      = {D_A}")
    print(f"dx, dy   = {DX}, {DY}")
    print(f"Omega    = {OMEGA}")
    print(f"h        = {H:.8e}")
    print(f"epsilon  = {EPS:.8e}")

    all_results = []

    seed = SEED

    # --------------------------------------------------------------
    # Normal small-dt accuracy sweep
    # --------------------------------------------------------------
    for case_name, state in CASES.items():

        for dt in DT_VALUES:

            result = run_case(
                case_name=case_name,
                state=state,
                dt=dt,
                seed=seed,
            )

            all_results.append(result)
            seed += 10

    # --------------------------------------------------------------
    # Deliberate boundary stress tests
    #
    # These are not intended as production dt values.
    # They just make the Gaussian boundary pathology visible.
    # --------------------------------------------------------------
    for case_name in (
        "one_quantum_A",
        "one_quantum_vacancy",
    ):

        result = run_case(
            case_name=case_name + "_STRESS",
            state=CASES[case_name],
            dt=STRESS_DT,
            seed=seed,
        )

        all_results.append(result)
        seed += 10

    # --------------------------------------------------------------
    # Compact final table
    # --------------------------------------------------------------
    print()
    print()
    print("=" * 110)
    print("COMPACT SUMMARY")
    print("=" * 110)

    print(
        f"{'case':30s} "
        f"{'dt':>8s} "
        f"{'old var/th':>12s} "
        f"{'new var/th':>12s} "
        f"{'old viol':>12s} "
        f"{'new viol':>12s}"
    )

    for r in all_results:

        print(
            f"{r['case']:30s} "
            f"{r['dt']:8.4g} "
            f"{r['old']['var_ratio']:12.5f} "
            f"{r['new']['var_ratio']:12.5f} "
            f"{r['old']['violation_rate']:12.4e} "
            f"{r['new']['violation_rate']:12.4e}"
        )

    # --------------------------------------------------------------
    # Plot variance accuracy
    # --------------------------------------------------------------
    fig, ax = plt.subplots()

    for case_name in CASES:

        subset = [
            r for r in all_results
            if r["case"] == case_name
        ]

        if not subset:
            continue

        dts = np.array([r["dt"] for r in subset])
        ratios = np.array([
            r["new"]["var_ratio"]
            for r in subset
        ])

        ax.plot(
            dts,
            ratios,
            marker="o",
            label=case_name,
        )

    ax.axhline(1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(
        "pure-death empirical variance / "
        "small-dt theoretical variance"
    )
    ax.set_title("Variance convergence of pure-death face sampler")
    ax.legend()
    fig.tight_layout()

    # --------------------------------------------------------------
    # Plot Gaussian vs reaction increment histogram for interior
    # at the largest normal dt.
    # --------------------------------------------------------------
    target = None

    for r in all_results:
        if (
            r["case"] == "interior_homogeneous"
            and r["dt"] == max(DT_VALUES)
        ):
            target = r
            break

    if target is not None:

        fig, ax = plt.subplots()

        d_old = target["old_aL"] - target["aL0"]
        d_new = target["new_aL"] - target["aL0"]

        ax.hist(
            d_old,
            bins=80,
            density=True,
            histtype="step",
            label="Gaussian FE",
        )

        ax.hist(
            d_new,
            bins=80,
            density=True,
            histtype="step",
            label="pure-death",
        )

        ax.set_xlabel(r"$\Delta\rho_L^A$")
        ax.set_ylabel("density")
        ax.set_title(
            "One-face increment distribution\n"
            f"interior homogeneous, dt={target['dt']}"
        )
        ax.legend()

        fig.tight_layout()
    plt.savefig('benchmark_schelling_face_sampler.png')
    plt.show()


if __name__ == "__main__":
    main()