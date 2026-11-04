"""Regression tests for the beta=0 canonical reference and full-lag FFT."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fhd.observables import (
    beta0_reference_covariance, connected_time_lag_covariance,
    excess_spectral_power, compute_observables, spatial_structure_factor,
)
from scripts.plot_schelling_phase_diagram import temporal_diagnostic_summary


@pytest.mark.parametrize('bc', ['Neumann', 'periodic'])
@pytest.mark.parametrize('demean_time', [False, True])
def test_all_fft_lags_equal_direct(bc, demean_time):
    rng = np.random.default_rng(122)
    phi = rng.normal(.35, .03, (2, 18, 9, 10))
    kw = dict(dt=.12, L=(50,50), bc=bc, n_bins=12,
              burn_in=3, lags='all', demean_time=demean_time)
    fast = connected_time_lag_covariance(phi, method='fft',
                                         mode_chunk_size=7, **kw)
    slow = connected_time_lag_covariance(phi, method='direct', **kw)
    np.testing.assert_allclose(fast['C'], slow['C'], rtol=1e-10,
                               atol=1e-12, equal_nan=True)
    assert fast['C'].shape[0] == phi.shape[1]-3
    assert fast['n_frames_used'] == phi.shape[1]-3


def test_canonical_beta0_reference():
    rng = np.random.default_rng(83)
    nx, ny, nA, nB = 8, 8, 16, 24
    M, T = nx*ny, 1200
    phi = np.zeros((2,T,nx,ny))
    for t in range(T):
        perm = rng.permutation(M)
        phi[0,t].reshape(-1)[perm[:nA]] = 1.
        phi[1,t].reshape(-1)[perm[nA:nA+nB]] = 1.
    theory = beta0_reference_covariance(
        [nA/M,nB/M], cell_capacity=1, n_cells=M)
    S = spatial_structure_factor(phi, L=(8,8), bc='Neumann', n_bins=5)['S']
    empirical = np.nanmean(S, axis=(0,1)).real
    np.testing.assert_allclose(empirical, theory, atol=.015)
    baseline = excess_spectral_power(
        phi, L=(8,8), bc='Neumann', reference='beta0', n_bins=5)['baseline']
    np.testing.assert_allclose(baseline, np.broadcast_to(theory,baseline.shape), atol=1e-12)


def test_compute_observables_full_lags_and_debug():
    rng = np.random.default_rng(23)
    phi = rng.normal(.35, .01, (2,20,8,8))
    r = compute_observables(phi, L=(50,50), dt_saved=.2, bc='Neumann',
                            reference='beta0', burn_in=5, lags='all',
                            n_bins=8)
    assert r['time_lag']['C'].shape[0] == 15
    assert r['time_lag']['demean_time'] is False
    assert r['time_lag']['method'] == 'fft'
    dynamic = dict(k_dynamic=r['time_lag']['k'],
                   tau=r['time_lag']['tau'],
                   C_k_tau=r['time_lag']['C'],
                   omega=r['spatiotemporal']['omega'],
                   S_k_omega=r['spatiotemporal']['S'])
    debug = temporal_diagnostic_summary(dynamic)
    assert debug['n_saved_lags'] == 15
    assert np.isclose(debug['post_burn_duration'], 15*.2)


def test_complex_mode_growth_and_frequency_use_same_band():
    from fhd.observables import eigenvalues_from_covariance
    dt = .1
    k = np.arange(.55, 2.55, .1)
    cov = np.zeros((2, len(k), 2, 2))
    cov[0] = np.eye(2)
    # A small-k oscillatory mode grows slowly, while a larger-k mode
    # oscillates faster but is damped. The independent maxima differ.
    for i, kval in enumerate(k):
        r, om = ((.004, .08) if kval < 1.1 else (-.3, 1.1))
        a = np.exp(r*dt)
        c, s = np.cos(om*dt), np.sin(om*dt)
        cov[1,i] = a * np.array([[c,-s],[s,c]])
    result = eigenvalues_from_covariance(
        k, [0,dt], cov, counts=np.full(len(k),20),
        k_range=(.5,2.5), widths=(.35,.5), step=.1,
        min_cycles=0, duration=200)
    assert np.isclose(result['Omega_max'],1.1,atol=1e-7)
    assert np.isclose(result['r_complex_max'],.004,atol=1e-7)
    assert np.isclose(result['Omega_at_r_complex_max'],.08,atol=1e-7)
    assert result['r_at_Omega_max'] < 0
