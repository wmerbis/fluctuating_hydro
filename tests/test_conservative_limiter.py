import pytest

np = pytest.importorskip("numpy")

from finite_volume_stochastic import ModelParameters, SchellingVoterFVSolver, make_random_initial_condition


def _params(nx, *, D_v=0.05):
    return ModelParameters(
        D_A=0.1,
        D_B=0.12,
        D_v=D_v,
        beta=6.0,
        h_noise=1.0 / nx,
        kappa=np.array([[0.6, -0.4], [-0.3, 0.5]], dtype=float),
        gamma=np.eye(2),
    )


def _solver(mode, nx=12, ny=10, seed=1, *, stochastic=False, D_v=0.05):
    return SchellingVoterFVSolver(
        nx=nx,
        ny=ny,
        lx=1.0,
        ly=1.0,
        params=_params(nx, D_v=D_v),
        bc="periodic",
        semi_implicit_stiff=False,
        stochastic=stochastic,
        limiter_mode=mode,
        rng=np.random.default_rng(seed),
        use_periodic_fast_path=True,
    )


def _drift(after, before, key):
    return after[key] - before[key]


def _assert_simplex(solver, tol=1.0e-12):
    assert float(np.min(solver.rho_A)) >= -tol
    assert float(np.min(solver.rho_B)) >= -tol
    assert float(np.max(solver.rho_A + solver.rho_B)) <= 1.0 + tol


def test_none_without_projection_events_conserves_roundoff():
    solver = _solver("none", stochastic=False, D_v=0.0)
    rho_A0, rho_B0 = make_random_initial_condition(12, 10, rhoA0=0.28, rhoB0=0.24, noise=1.0e-3, seed=3)
    solver.set_state(rho_A0, rho_B0)
    before = solver.total_masses()
    for _ in range(8):
        solver.step(2.0e-5, add_noise=False)
    after = solver.total_masses()
    assert abs(_drift(after, before, "A")) < 1.0e-14
    assert abs(_drift(after, before, "B")) < 1.0e-14
    assert abs(_drift(after, before, "occupied")) < 1.0e-14
    assert abs(_drift(after, before, "total")) < 1.0e-14


def test_clip_projection_can_create_occupied_mass_drift():
    solver = _solver("clip", nx=4, ny=3)
    rho_A = np.full((3, 4), 0.45)
    rho_B = np.full((3, 4), 0.45)
    solver.set_state(rho_A, rho_B)
    before = solver.total_masses()

    # Mimic a projection-triggering update: the negative value is clipped locally
    # and the overfull cell is rescaled locally, with no conservative neighbor
    # compensation.  This documents the drift mechanism that remains available
    # for comparison under limiter_mode="clip".
    solver.rho_A[0, 0] = -0.12
    solver.rho_B[0, 0] = 0.50
    solver.rho_A[1, 1] = 0.80
    solver.rho_B[1, 1] = 0.70
    pre_projection = solver.total_masses()
    solver.apply_simplex_limiter()
    after = solver.total_masses()

    assert abs(_drift(after, pre_projection, "occupied")) > 1.0e-4
    assert abs(_drift(after, before, "occupied")) > 1.0e-4
    _assert_simplex(solver)


def test_conservative_limiter_preserves_occupied_mass_and_bounds_when_active():
    nx = ny = 10
    solver = _solver("conservative", nx=nx, ny=ny, seed=11, stochastic=True, D_v=0.08)
    rho_A0, rho_B0 = make_random_initial_condition(nx, ny, rhoA0=0.47, rhoB0=0.47, noise=4.0e-3, seed=5)
    solver.set_state(rho_A0, rho_B0)
    before = solver.total_masses()
    activations = 0
    for _ in range(20):
        solver.step(4.0e-4, add_noise=True)
        activations += int(solver.last_limiter_stats["activations"])
        _assert_simplex(solver, tol=5.0e-12)
    after = solver.total_masses()

    assert activations > 0
    assert abs(_drift(after, before, "occupied")) < 5.0e-13
    assert abs(_drift(after, before, "total")) < 5.0e-13


def test_conservative_limiter_substantially_reduces_clip_occupied_drift():
    nx = ny = 10
    rho_A0, rho_B0 = make_random_initial_condition(nx, ny, rhoA0=0.48, rhoB0=0.48, noise=5.0e-3, seed=12)
    clip = _solver("clip", nx=nx, ny=ny, seed=21, stochastic=True, D_v=0.04)
    conservative = _solver("conservative", nx=nx, ny=ny, seed=21, stochastic=True, D_v=0.04)
    clip.set_state(rho_A0, rho_B0)
    conservative.set_state(rho_A0, rho_B0)
    clip_before = clip.total_masses()
    conservative_before = conservative.total_masses()
    for _ in range(25):
        clip.step(5.0e-4, add_noise=True)
        conservative.step(5.0e-4, add_noise=True)

    clip_drift = abs(_drift(clip.total_masses(), clip_before, "occupied"))
    conservative_drift = abs(_drift(conservative.total_masses(), conservative_before, "occupied"))
    assert conservative_drift < 5.0e-13
    assert clip_drift > 10.0 * conservative_drift + 1.0e-8
    _assert_simplex(clip)
    _assert_simplex(conservative)


def test_conservative_limiter_reports_flux_and_theta_diagnostics():
    nx = ny = 8
    solver = _solver("conservative", nx=nx, ny=ny, seed=33, stochastic=True, D_v=0.08)
    rho_A0, rho_B0 = make_random_initial_condition(nx, ny, rhoA0=0.48, rhoB0=0.47, noise=4.0e-3, seed=6)
    solver.set_state(rho_A0, rho_B0)
    solver.step(5.0e-4, add_noise=True)
    stats = solver.last_limiter_stats

    assert stats["mode"] == "conservative"
    assert stats["limited_faces"] == stats["limited_faces_x"] + stats["limited_faces_y"]
    assert 0.0 <= stats["theta_min"] <= stats["theta_mean"] <= stats["theta_max"] <= 1.0
    assert 0.0 <= stats["theta_frac_lt_1"] <= 1.0
    assert 0.0 <= stats["theta_frac_lt_0_5"] <= stats["theta_frac_lt_1"]
    assert 0.0 <= stats["theta_frac_lt_0_1"] <= stats["theta_frac_lt_0_5"]
    for prefix in ("A", "B", "occupied", "deterministic_occupied", "schelling_noise_occupied"):
        assert stats[f"flux_{prefix}_l1_raw"] >= stats[f"flux_{prefix}_l1_limited"] >= 0.0
        assert stats[f"flux_{prefix}_l2_raw"] >= stats[f"flux_{prefix}_l2_limited"] >= 0.0
        assert 0.0 <= stats[f"flux_{prefix}_removed_fraction"] <= 1.0
    for key in ("limiting_corr_rho_A", "limiting_corr_rho_B", "limiting_corr_rho_0", "limiting_corr_grad_rho"):
        assert -1.0 <= stats[key] <= 1.0
