import numpy as np
import time
from scipy.stats import ks_2samp, wasserstein_distance

from fhd.FHD_2D import fhd_2d

# ============================================================
# SETTINGS
# ============================================================

TAU = 0.005

# Transition samples per p0.
# Start with 300_000--500_000.
# Increase to ~2e6 for a stricter comparison.
N_TRANS = 500_000

# Samples used specifically for comparing M.
N_M = 2_000_000

SEED = 12345

P_VALUES = np.array([
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    5e-2,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.998,
    0.999,
    0.9995,
])


# ============================================================
# CREATE TWO SOLVERS
#
# These spatial parameters are irrelevant here because we call
# the local WF transition kernel directly.
# ============================================================

common_kwargs = dict(
    L=(1.0, 1.0),
    N=(8, 8),
    bc="periodic",
    voter_noise_mode="wright_fisher",
    wf_target_digits=26,
)

# Force exact q_m branch:
#
# condition is:
#     tau < wf_gaussian_threshold -> Griffiths
#
# threshold=0 means tau=0.002 always goes exact.
wf_exact = fhd_2d(
    **common_kwargs,
    wf_gaussian_threshold=0.0,
)

# Force Griffiths branch for the same tau.
wf_griff = fhd_2d(
    **common_kwargs,
    wf_gaussian_threshold=np.inf,
)

wf_exact.set_seed(SEED)
wf_griff.set_seed(SEED + 1)


# ============================================================
# 1. BUILD EXACT q_m TABLE
# ============================================================

print("=" * 72)
print(f"Benchmarking exact WF vs Griffiths at tau = {TAU}")
print("=" * 72)

t0 = time.perf_counter()

table = wf_exact._get_wf_qm_table(TAU)

table_time = time.perf_counter() - t0

q_exact = np.asarray(table.q, dtype=np.float64)
m_values = np.arange(1, len(q_exact) + 1)

print()
print("Exact q_m table")
print("----------------")
print(f"m range:        1 ... {len(q_exact)}")
print(f"sum q_m:        {q_exact.sum():.17g}")
print(f"build time:     {table_time:.3f} s")


# ============================================================
# 2. COMPARE THE AUXILIARY LINEAGE COUNT M
#
# This is the most direct test of the Griffiths approximation.
# ============================================================

print()
print("=" * 72)
print("LINEAGE COUNT M")
print("=" * 72)

# Exact M
t0 = time.perf_counter()

M_exact = wf_exact._sample_wf_lineage_count(
    TAU,
    N_M,
)

t_exact_M = time.perf_counter() - t0


# Griffiths M
t0 = time.perf_counter()

M_griff = wf_griff._sample_wf_lineage_count(
    TAU,
    N_M,
)

t_griff_M = time.perf_counter() - t0


# Griffiths target moments
mu_g, var_g = wf_griff._wf_griffiths_moments_zero(TAU)


# Exact q_m moments from table
mu_q = np.sum(m_values * q_exact)
var_q = np.sum(
    (m_values - mu_q)**2 * q_exact
)


# Empirical moments
mu_exact_emp = M_exact.mean()
var_exact_emp = M_exact.var()

mu_griff_emp = M_griff.mean()
var_griff_emp = M_griff.var()


# ------------------------------------------------------------
# Heterozygosity contraction factor
#
# Exact WF identity:
#
# E[(M-1)/(M+1)] = exp(-tau)
# ------------------------------------------------------------

h_theory = np.exp(-TAU)

h_q = np.sum(
    q_exact
    * (m_values - 1.0)
    / (m_values + 1.0)
)

h_exact_emp = np.mean(
    (M_exact - 1.0)
    / (M_exact + 1.0)
)

h_griff_emp = np.mean(
    (M_griff - 1.0)
    / (M_griff + 1.0)
)


# ------------------------------------------------------------
# Empirical Griffiths PMF vs exact q_m
#
# Total variation distance:
#
# TV = 1/2 sum_m |p_m - q_m|
# ------------------------------------------------------------

max_M = max(
    int(M_griff.max()),
    len(q_exact),
)

counts_g = np.bincount(
    M_griff,
    minlength=max_M + 1,
)

pmf_g = counts_g / N_M

q_pad = np.zeros(max_M + 1)
q_pad[1:len(q_exact) + 1] = q_exact

tv_M = 0.5 * np.sum(
    np.abs(pmf_g - q_pad)
)


# A KS test is also useful, although TV is more natural here.
ks_M = ks_2samp(
    M_exact,
    M_griff,
)


print(f"""
Exact q_m moments
    E[M]              = {mu_q:.10f}
    Var[M]            = {var_q:.10f}

Exact sampled M
    E[M]              = {mu_exact_emp:.10f}
    Var[M]            = {var_exact_emp:.10f}

Griffiths formula
    E[M]              = {mu_g:.10f}
    Var[M]            = {var_g:.10f}

Griffiths sampled M
    E[M]              = {mu_griff_emp:.10f}
    Var[M]            = {var_griff_emp:.10f}

Heterozygosity factor
    exp(-tau)         = {h_theory:.14e}
    exact q_m         = {h_q:.14e}
    exact samples     = {h_exact_emp:.14e}
    Griffiths samples = {h_griff_emp:.14e}

Distribution comparison
    TV distance       = {tv_M:.6e}
    KS statistic      = {ks_M.statistic:.6e}

Timing
    exact M sampling  = {t_exact_M:.3f} s
    Griffiths M       = {t_griff_M:.3f} s
""")


# ============================================================
# 3. EXACT FINITE-TIME ABSORPTION PROBABILITIES
#
# Given M:
#
# P(p'=0 | M=m) = (1-p)^m
# P(p'=1 | M=m) = p^m
#
# therefore average over exact q_m.
# ============================================================

def exact_absorption_probabilities(p0, q, m):
    p0_zero = np.sum(
        q * np.power(1.0 - p0, m)
    )

    p0_one = np.sum(
        q * np.power(p0, m)
    )

    return p0_zero, p0_one


# ============================================================
# 4. COMPARE FULL WF TRANSITION DISTRIBUTIONS
# ============================================================

print()
print("=" * 72)
print("WRIGHT-FISHER TRANSITIONS")
print("=" * 72)

header = (
    "p0       "
    "mean(ex)   mean(G)    "
    "var(th)      var(ex)      var(G)       "
    "P0(th)     P0(ex)     P0(G)      "
    "P1(th)     P1(ex)     P1(G)      "
    "KS         Wasserstein"
)

print(header)
print("-" * len(header))


all_results = []

for j, p0 in enumerate(P_VALUES):

    # Separate deterministic seeds per p0 so one case doesn't
    # affect the random stream of the next one.
    wf_exact.set_seed(SEED + 1000 + 2*j)
    wf_griff.set_seed(SEED + 1001 + 2*j)

    p_input = np.full(
        N_TRANS,
        p0,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Exact q_m transition
    # --------------------------------------------------------

    t0 = time.perf_counter()

    p_exact = wf_exact._sample_wright_fisher_transition(
        p_input,
        TAU,
    )

    time_exact = time.perf_counter() - t0


    # --------------------------------------------------------
    # Griffiths M transition
    # --------------------------------------------------------

    t0 = time.perf_counter()

    p_griff = wf_griff._sample_wright_fisher_transition(
        p_input,
        TAU,
    )

    time_griff = time.perf_counter() - t0


    # --------------------------------------------------------
    # Exact theoretical first two moments
    # --------------------------------------------------------

    mean_theory = p0

    var_theory = (
        p0
        * (1.0 - p0)
        * (1.0 - np.exp(-TAU))
    )


    # Exact finite-time boundary masses from q_m
    P0_theory, P1_theory = exact_absorption_probabilities(
        p0,
        q_exact,
        m_values,
    )


    # --------------------------------------------------------
    # Empirical statistics
    # --------------------------------------------------------

    mean_exact = p_exact.mean()
    mean_griff = p_griff.mean()

    var_exact = p_exact.var()
    var_griff = p_griff.var()

    P0_exact = np.mean(p_exact == 0.0)
    P1_exact = np.mean(p_exact == 1.0)

    P0_griff = np.mean(p_griff == 0.0)
    P1_griff = np.mean(p_griff == 1.0)


    # --------------------------------------------------------
    # Distribution distances
    # --------------------------------------------------------

    ks = ks_2samp(
        p_exact,
        p_griff,
    ).statistic

    wasserstein = wasserstein_distance(
        p_exact,
        p_griff,
    )


    all_results.append({
        "p0": p0,

        "mean_theory": mean_theory,
        "mean_exact": mean_exact,
        "mean_griff": mean_griff,

        "var_theory": var_theory,
        "var_exact": var_exact,
        "var_griff": var_griff,

        "P0_theory": P0_theory,
        "P0_exact": P0_exact,
        "P0_griff": P0_griff,

        "P1_theory": P1_theory,
        "P1_exact": P1_exact,
        "P1_griff": P1_griff,

        "KS": ks,
        "wasserstein": wasserstein,

        "time_exact": time_exact,
        "time_griff": time_griff,
    })


    print(
        f"{p0:<8.4g} "
        f"{mean_exact:10.6f} "
        f"{mean_griff:10.6f} "
        f"{var_theory:11.4e} "
        f"{var_exact:11.4e} "
        f"{var_griff:11.4e} "
        f"{P0_theory:10.3e} "
        f"{P0_exact:10.3e} "
        f"{P0_griff:10.3e} "
        f"{P1_theory:10.3e} "
        f"{P1_exact:10.3e} "
        f"{P1_griff:10.3e} "
        f"{ks:9.2e} "
        f"{wasserstein:11.3e}"
    )


# ============================================================
# 5. SUMMARY OF RELATIVE ERRORS
# ============================================================

print()
print("=" * 72)
print("MAXIMUM OBSERVED DIFFERENCES")
print("=" * 72)

var_rel_errors = []

mean_diffs = []

for r in all_results:

    if r["var_theory"] > 0:
        var_rel_errors.append(
            abs(
                r["var_griff"]
                - r["var_theory"]
            )
            / r["var_theory"]
        )

    mean_diffs.append(
        abs(
            r["mean_griff"]
            - r["mean_theory"]
        )
    )

print(
    "max |mean_G - p0|        =",
    f"{max(mean_diffs):.6e}"
)

print(
    "max relative variance err =",
    f"{max(var_rel_errors):.6e}"
)

print(
    "max KS(WF, Griffiths)     =",
    f"{max(r['KS'] for r in all_results):.6e}"
)

print(
    "max Wasserstein           =",
    f"{max(r['wasserstein'] for r in all_results):.6e}"
)

print()
print("Mean sampling times per p0:")

print(
    "exact q_m:",
    np.mean([
        r["time_exact"]
        for r in all_results
    ])
)

print(
    "Griffiths:",
    np.mean([
        r["time_griff"]
        for r in all_results
    ])
)