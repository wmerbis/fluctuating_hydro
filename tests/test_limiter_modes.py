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
    assert solver.last_limiter_stats["mode"] == "none"
    assert solver.last_limiter_stats["limited_cells"] == 0


def test_clip_projection_can_create_occupied_mass_drift():
    solver = _solver("clip", nx=4, ny=3)
    rho_A = np.full((3, 4), 0.45)
    rho_B = np.full((3, 4), 0.45)
    solver.set_state(rho_A, rho_B)
    before = solver.total_masses()

    # Mimic a projection-triggering update: the negative value is clipped locally
    # and the overfull cell is rescaled locally, with no conservative neighbor
    # compensation. This documents the drift mechanism for limiter_mode="clip".
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


def test_clip_step_reports_projection_diagnostics():
    solver = _solver("clip", nx=4, ny=3)
    solver.set_state(np.full((3, 4), 0.48), np.full((3, 4), 0.48))
    before_A = solver.rho_A.copy()
    before_B = solver.rho_B.copy()
    solver._rho_A_star[...] = before_A
    solver._rho_B_star[...] = before_B
    solver._rho_A_star[0, 0] = -0.1
    solver._rho_B_star[0, 0] = 0.5
    pre_A = float(np.sum(solver._rho_A_star) * solver.vol)
    pre_B = float(np.sum(solver._rho_B_star) * solver.vol)
    corrections = solver.apply_simplex_limiter(solver._rho_A_star, solver._rho_B_star)
    post_A = float(np.sum(solver._rho_A_star) * solver.vol)
    post_B = float(np.sum(solver._rho_B_star) * solver.vol)

    assert corrections == 1
    assert post_A - pre_A > 0.0
    assert post_B == pytest.approx(pre_B)


def test_removed_limiter_modes_are_rejected():
    for mode in ("conservative", "conservative_old", "conservative_fct", "fct"):
        with pytest.raises(ValueError, match="limiter_mode must be 'none' or 'clip'"):
            _solver(mode)


def test_run_returns_only_clip_relevant_limiter_series():
    solver = _solver("clip", nx=8, ny=8, stochastic=True)
    rho_A0, rho_B0 = make_random_initial_condition(8, 8, rhoA0=0.45, rhoB0=0.45, noise=1.0e-3, seed=5)
    solver.set_state(rho_A0, rho_B0)
    out = solver.run(1.0e-5, 2, record_masses_every=1, add_noise=True)

    assert out["limiter_mode"] == "clip"
    assert "limiter_limited_cells" in out
    assert "limiter_projection_delta_occupied" in out
    assert "limiter_theta_mean" not in out
    assert "limiter_limited_faces" not in out
