#!/usr/bin/env python3
"""
Compare a machine-precision implementation of the exact neutral Wright-Fisher
transition law against the DCM nearest-absorbing-boundary approximation.

Exact local stochastic problem
------------------------------
    dp = sqrt(p(1-p)) dW_tau,     0 <= p <= 1

where tau is the dimensionless stochastic time.  For the Schelling-Voter
FHD discretization discussed in this project,

    tau = noise_strength**2 * 4*d*D_v*dt / (dx*dy)

(with d=2 in two spatial dimensions).

The exact transition law is the coalescent mixture

    M ~ q_m^0(tau)
    L | M ~ Binomial(M, p)
    p' = 0                         if L = 0
       = 1                         if L = M
       ~ Beta(L, M-L)             otherwise.

For theta=0, q_m^0(tau) is the entrance-law distribution of the number of
surviving lineages in Kingman's coalescent, whose death rates are
m(m-1)/2.  We precompute q_m^0(tau) from its alternating series using mpmath
arbitrary precision, then convert the final PMF/CDF to float64.  Thus the
runtime sampler follows the exact transition law up to the requested table
truncation / floating-point tolerance; it is not a formally retrospective
'exact algorithm' in the Jenkins-Spano sense.

The DCM comparison uses
    dm = sqrt(1-|m|) dW_tau,   m = 2p-1,
chooses the closest absorbing boundary, samples the corresponding one-sided
square-root process with Poisson-Gamma, and clips any far-boundary crossing
back to [-1,1], matching the diagnostic implementation discussed here.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.special import gammaln


# -----------------------------------------------------------------------------
# Wright-Fisher / FHD time conversion
# -----------------------------------------------------------------------------

def voter_tau(
    D_v: float,
    dt: float,
    dx: float,
    dy: float,
    noise_strength: float = 1.0,
    d: int = 2,
) -> float:
    """Dimensionless WF time for one demographic-noise substep."""
    return (
        noise_strength**2
        * 4.0
        * d
        * D_v
        * dt
        / (dx * dy)
    )


# -----------------------------------------------------------------------------
# q_m^0(tau): entrance law of the mutation-free Kingman lineage-count process
# -----------------------------------------------------------------------------

@dataclass
class QMTable:
    tau: float
    q: np.ndarray          # q[j] = P(M = j+1)
    cdf: np.ndarray
    raw_sum: float
    clipped_negative_mass: float
    m_max: int
    build_seconds: float


def _estimate_peak_log10_term(m: int, tau: float) -> float:
    """
    Estimate the largest log10 magnitude in the alternating q_m series.

    This determines how many decimal digits are needed to survive cancellation.
    Uses float64 log arithmetic only for choosing mpmath precision.
    """
    log_b = (
        math.log(2 * m - 1)
        + gammaln(2 * m - 1)
        - gammaln(m)
        - gammaln(m + 1)
        - 0.5 * m * (m - 1) * tau
    )
    max_log_b = log_b
    k = m

    # The term ratio is eventually monotone below unity.  We only need the peak.
    for _ in range(1_000_000):
        log_ratio = (
            math.log(2 * k + 1)
            - math.log(2 * k - 1)
            + math.log(m + k - 1)
            - math.log(k + 1 - m)
            - k * tau
        )
        log_b += log_ratio
        max_log_b = max(max_log_b, log_b)
        k += 1
        if log_ratio < 0.0:
            break
    else:
        raise RuntimeError("Could not locate decreasing tail of q_m series")

    return max_log_b / math.log(10.0)


def q_m_zero_mutation_mp(
    m: int,
    tau: float,
    target_digits: int = 26,
) -> float:
    r"""
    Compute q_m^0(tau) from

      q_m(tau) = sum_{k=m}^inf (-1)^(k-m) a_{k,m}
                 exp[-k(k-1) tau / 2],

      a_{k,m} = (2k-1) (m)_{k-1} / [m! (k-m)!].

    The terms are generated recursively to avoid repeated Gamma evaluations.
    Arbitrary precision is selected adaptively from the estimated peak term.    
    """
    if m < 1:
        raise ValueError("For theta=0, M has support m >= 1")
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    peak_log10 = _estimate_peak_log10_term(m, tau)
    dps = max(
        50,
        int(math.ceil(max(0.0, peak_log10))) + target_digits + 20,
    )

    with mp.workdps(dps):
        t = mp.mpf(str(float(tau)))

        # First positive term k=m.
        b = (
            (2 * m - 1)
            * mp.gamma(2 * m - 1)
            / (mp.gamma(m) * mp.gamma(m + 1))
            * mp.exp(-mp.mpf(m * (m - 1)) * t / 2)
        )

        total = b
        previous = b
        sign = -1
        k = m
        decreasing = False

        # Absolute error target for an individual probability.  This is far below
        # the precision needed for the final float64 CDF.
        term_tol = mp.power(10, -(target_digits + 6))

        for _ in range(1_000_000):
            # b_{k+1}/b_k for theta=0.
            ratio = (
                (mp.mpf(2 * k + 1) / (2 * k - 1))
                * (mp.mpf(m + k - 1) / (k + 1 - m))
                * mp.exp(-mp.mpf(k) * t)
            )
            b *= ratio
            k += 1

            if b < previous:
                decreasing = True

            total += sign * b
            sign = -sign

            # Once the alternating terms decrease, the first omitted term bounds
            # the truncation error.
            if decreasing and b < term_tol:
                return float(total)

            previous = b

    raise RuntimeError(f"q_m series failed to converge for m={m}, tau={tau}")


def build_qm_table(
    tau: float,
    target_digits: int = 26,
    tail_sigma: float = 12.0,
    tail_padding: int = 20,
    pmf_tolerance: float = 5e-14,
) -> QMTable:
    """
    Precompute q_m^0(tau), m=1,...,m_max, and its float64 CDF.

    For small tau the lineage count is concentrated around
        mean ~ 2/tau,
        variance ~ 2/(3 tau).
    We use that only to choose a generously large initial upper cutoff.  The
    probabilities themselves come from the exact alternating series.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    t0 = time.perf_counter()

    mu_asym = 2.0 / tau
    sigma_asym = math.sqrt(2.0 / (3.0 * tau))
    m_max = max(
        5,
        int(math.ceil(mu_asym + tail_sigma * sigma_asym + tail_padding)),
    )

    values = [
        q_m_zero_mutation_mp(m, tau, target_digits=target_digits)
        for m in range(1, m_max + 1)
    ]
    q = np.asarray(values, dtype=np.float64)

    # Arbitrary-precision cancellation can leave utterly negligible (~1e-30)
    # signed roundoff after conversion.  Refuse any materially negative PMF.
    materially_negative = q < -pmf_tolerance
    if np.any(materially_negative):
        bad = np.where(materially_negative)[0] + 1
        raise RuntimeError(
            "Materially negative q_m values; increase target_digits. "
            f"First bad m values: {bad[:10]}"
        )

    clipped_negative_mass = float(-q[q < 0.0].sum())
    q[q < 0.0] = 0.0

    raw_sum = float(q.sum())
    residual = 1.0 - raw_sum
    if abs(residual) > pmf_tolerance:
        raise RuntimeError(
            "q_m table does not sum sufficiently close to one: "
            f"sum={raw_sum:.17g}, residual={residual:.3e}. "
            "Increase tail_sigma/tail_padding and/or target_digits."
        )

    # Normalize only at float64 roundoff level so cdf[-1] is exactly one.
    q /= q.sum()
    cdf = np.cumsum(q)
    cdf[-1] = 1.0

    return QMTable(
        tau=tau,
        q=q,
        cdf=cdf,
        raw_sum=raw_sum,
        clipped_negative_mass=clipped_negative_mass,
        m_max=m_max,
        build_seconds=time.perf_counter() - t0,
    )


# -----------------------------------------------------------------------------
# Vectorized exact Wright-Fisher transition sampler
# -----------------------------------------------------------------------------

def sample_wright_fisher_exact(
    p: np.ndarray,
    table: QMTable,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized transition draw for dp=sqrt(p(1-p)) dW_tau.

    p may have any shape.  Exact absorbing states 0 and 1 are left unchanged.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must lie in [0,1]")

    out = p.copy()
    active = (p > 0.0) & (p < 1.0)
    if not np.any(active):
        return out

    pp = p[active]

    # M is independent of p, which is the key reason this can be cached/vectorized.
    u = rng.random(pp.size)
    M = np.searchsorted(table.cdf, u, side="right") + 1

    L = rng.binomial(M, pp)

    new = np.empty_like(pp)
    at_zero = L == 0
    at_one = L == M
    interior = ~(at_zero | at_one)

    new[at_zero] = 0.0
    new[at_one] = 1.0
    if np.any(interior):
        new[interior] = rng.beta(
            L[interior],
            M[interior] - L[interior],
        )

    out[active] = new
    return out


def wright_fisher_voter_step_phi(
    phi: np.ndarray,
    table: QMTable,
    rng: np.random.Generator,
    occupancy_tol: float = 0.0,
) -> np.ndarray:
    """
    Exact-WF demographic Voter substep for phi.shape == (2,Nx,Ny).

    Conserves n=rho_A+rho_B pointwise and requires no post-noise projection.
    Modifies phi in place and returns it.
    """
    A = phi[0]
    B = phi[1]
    n = A + B
    occupied = n > occupancy_tol

    p = np.zeros_like(n, dtype=np.float64)
    p[occupied] = A[occupied] / n[occupied]

    # Tiny input cleanup can be moved into your diagnostics when integrating.
    np.clip(p, 0.0, 1.0, out=p)
    p_new = sample_wright_fisher_exact(p, table, rng)

    A[occupied] = n[occupied] * p_new[occupied]
    B[occupied] = n[occupied] - A[occupied]
    A[~occupied] = 0.0
    B[~occupied] = 0.0
    return phi


# -----------------------------------------------------------------------------
# DCM nearest-barrier comparison sampler
# -----------------------------------------------------------------------------

def sample_dcm_nearest_barrier(
    p: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    repair_far_boundary: bool = True,
):
    """
    DCM nearest-barrier sampler in the same dimensionless time tau.

    Returns
    -------
    p_new : ndarray
    diagnostics : dict
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must lie in [0,1]")

    m = 2.0 * p - 1.0
    s = np.sign(m)

    # At m=0 the two barriers are equidistant; choose symmetrically.
    tie = s == 0.0
    if np.any(tie):
        s[tie] = np.where(
            rng.random(np.count_nonzero(tie)) < 0.5,
            -1.0,
            1.0,
        )

    y = 1.0 - s * m
    lam = 2.0 / tau

    Q = rng.poisson(lam * y)
    y_new = np.zeros_like(y)
    nonzero = Q > 0
    if np.any(nonzero):
        y_new[nonzero] = rng.gamma(
            shape=Q[nonzero],
            scale=1.0 / lam,
        )

    m_new = s * (1.0 - y_new)

    far_low = m_new < -1.0
    far_high = m_new > 1.0
    far = far_low | far_high

    if repair_far_boundary:
        np.clip(m_new, -1.0, 1.0, out=m_new)

    p_new = 0.5 * (1.0 + m_new)

    return p_new, {
        "n_far": int(np.count_nonzero(far)),
        "far_fraction": float(np.mean(far)),
        "n_far_low": int(np.count_nonzero(far_low)),
        "n_far_high": int(np.count_nonzero(far_high)),
    }


# -----------------------------------------------------------------------------
# Exact analytic diagnostics from the WF transition law
# -----------------------------------------------------------------------------

def wf_exact_mean(p0: float) -> float:
    return float(p0)


def wf_exact_variance(p0: float, tau: float) -> float:
    return float(p0 * (1.0 - p0) * (1.0 - math.exp(-tau)))


def wf_exact_absorption_probabilities(
    p0: float,
    table: QMTable,
) -> tuple[float, float]:
    """Finite-time exact P(P_tau=0) and P(P_tau=1) from the mixture."""
    m = np.arange(1, table.q.size + 1, dtype=np.float64)
    p_zero = float(np.sum(table.q * (1.0 - p0) ** m))
    p_one = float(np.sum(table.q * p0**m))
    return p_zero, p_one


# -----------------------------------------------------------------------------
# Comparison harness
# -----------------------------------------------------------------------------

def compare_one(
    p0: float,
    tau: float,
    n_samples: int,
    table: QMTable,
    seed: int,
):
    # Separate but identically seeded generators avoid call-order coupling.
    rng_wf = np.random.default_rng(seed)
    rng_dcm = np.random.default_rng(seed)
    x0 = np.full(n_samples, p0, dtype=np.float64)

    t0 = time.perf_counter()
    wf = sample_wright_fisher_exact(x0, table, rng_wf)
    wf_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    dcm, dcm_diag = sample_dcm_nearest_barrier(x0, tau, rng_dcm)
    dcm_seconds = time.perf_counter() - t0

    exact_p0, exact_p1 = wf_exact_absorption_probabilities(p0, table)

    return {
        "p0": p0,
        "tau": tau,
        "exact_mean_theory": wf_exact_mean(p0),
        "exact_var_theory": wf_exact_variance(p0, tau),
        "exact_abs0_theory": exact_p0,
        "exact_abs1_theory": exact_p1,
        "wf_mean": float(np.mean(wf)),
        "wf_var": float(np.var(wf)),
        "wf_abs0": float(np.mean(wf == 0.0)),
        "wf_abs1": float(np.mean(wf == 1.0)),
        "wf_seconds": wf_seconds,
        "dcm_mean": float(np.mean(dcm)),
        "dcm_var": float(np.var(dcm)),
        "dcm_abs0": float(np.mean(dcm == 0.0)),
        "dcm_abs1": float(np.mean(dcm == 1.0)),
        "dcm_far_fraction": dcm_diag["far_fraction"],
        "dcm_seconds": dcm_seconds,
    }


def print_comparison(rows):
    header = (
        "p0    mean(th) mean(WF) mean(DCM)   "
        "var(th)   var(WF)   var(DCM)   "
        "P0(th) P0(WF) P0(DCM)   "
        "P1(th) P1(WF) P1(DCM)   far(DCM)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['p0']:4.2f}  "
            f"{r['exact_mean_theory']:.6f} {r['wf_mean']:.6f} {r['dcm_mean']:.6f}   "
            f"{r['exact_var_theory']:.6e} {r['wf_var']:.6e} {r['dcm_var']:.6e}   "
            f"{r['exact_abs0_theory']:.3e} {r['wf_abs0']:.3e} {r['dcm_abs0']:.3e}   "
            f"{r['exact_abs1_theory']:.3e} {r['wf_abs1']:.3e} {r['dcm_abs1']:.3e}   "
            f"{r['dcm_far_fraction']:.3e}"
        )

    wf_time = sum(r["wf_seconds"] for r in rows)
    dcm_time = sum(r["dcm_seconds"] for r in rows)
    print()
    print(f"Total WF sampling time for table above:  {wf_time:.3f} s")
    print(f"Total DCM sampling time for table above: {dcm_time:.3f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Compare exact neutral Wright-Fisher and DCM one-step samplers."
    )
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--target-digits", type=int, default=26)
    args = parser.parse_args()

    print(f"Building q_m^0 table for tau={args.tau:g} ...")
    table = build_qm_table(
        args.tau,
        target_digits=args.target_digits,
    )
    print(
        f"m=1..{table.m_max}, build={table.build_seconds:.3f} s, "
        f"raw sum={table.raw_sum:.17g}, "
        f"clipped negative mass={table.clipped_negative_mass:.3e}"
    )
    print()

    rows = [
        compare_one(
            p0=p0,
            tau=args.tau,
            n_samples=args.n,
            table=table,
            seed=args.seed + i,
        )
        for i, p0 in enumerate(args.p_values)
    ]
    print_comparison(rows)


if __name__ == "__main__":
    main()
