"""Analytical linear-noise spectrum of the new Schelling hydrodynamics.

Standalone support module: only NumPy and SciPy required. Can be placed in
schelling_spectrum_fv2.py (at repository root) or fhd/schelling_spectrum_fv2.py.

Sources: October 2026 manuscript, SI Eqs. (S47), (S73), (S90),
(S118)-(S121), (S142)-(S143), (S152). Voter is OPTIONAL (D_v=0 default).

For periodic grids, the default --symbol fv2 matches the second-order
finite-volume face differences and second-order cell-centered Laplacian.
Neumann grids require a cosine/eigenmode formulation, not this periodic FFT.
Stable stationary covariance solves M C + C M.T = N; finite-time covariance
is evaluated for unstable modes too. This is linear Gaussian theory, NOT an
exact prediction of finite-time discrete face reactions near density bounds.
"""
from __future__ import annotations
import argparse
from math import comb
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm


def exact_theta(rho, beta, kappa):
    """Exact finite-state response Theta[a,b] (S47), without importing FHD_2D."""
    ra, rb = np.asarray(rho, dtype=float)
    r0 = 1.0 - ra - rb
    if min(ra, rb, r0) < -1e-12:
        raise ValueError('rho must be in the three-state simplex')
    kappa = np.asarray(kappa, dtype=float)
    if kappa.shape != (2, 2):
        raise ValueError('kappa must have shape (2,2)')
    th = np.zeros((2, 2), dtype=float)
    for ma in range(3):
        for mb in range(3 - ma):
            wm = comb(2, ma) * comb(2-ma, mb)
            for na in range(4):
                for nb in range(4 - na):
                    wn = comb(3, na) * comb(3-na, nb)
                    w = wm * wn * ra**(ma+na) * rb**(mb+nb) * r0**(5-ma-mb-na-nb)
                    for a in range(2):
                        x = 0.5 * beta * ((ma-na)*kappa[a, 0] + (mb-nb)*kappa[a, 1])
                        for b in range(2):
                            th[a, b] += w * (np.tanh(x + 0.5*beta*kappa[a, b]) - np.tanh(x))
    return th


def linear_matrices(rho=(0.35, 0.35), D=(1., 1.), beta=5.,
                    kappa=((0.6, -0.4), (-0.4, 0.6)), h=0.001, D_v=0.):
    """Return Theta, Gamma, A, Q, NS, NV, using SI conventions.

    ∂t δrho = A ∇²δrho + Q ∇⁴δrho + noise.
    <N(k,t)N(-k,t')^T> = (NV + q² NS) δ(t-t').
    """
    rho, D = np.asarray(rho, dtype=float), np.asarray(D, dtype=float)
    if rho.shape != (2,) or D.shape != (2,) or np.any(D < 0):
        raise ValueError('rho and D must both have shape (2,) and D>=0')
    rho0 = 1.0 - rho.sum()
    theta = exact_theta(rho, beta, kappa)
    gamma = 2.0 * h*h * theta
    A = np.empty((2, 2), dtype=float)
    Q = np.empty((2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            A[a, b] = D[a] * ((rho0 if a == b else 0.0) + rho[a] - 10.0*rho[a]*rho0*theta[a, b])
            Q[a, b] = -D[a] * rho[a]*rho0*gamma[a, b]
    A += D_v * np.array([[rho[1], -rho[0]], [-rho[1], rho[0]]])
    NS = np.diag(2.0*D*h*h*rho*rho0)
    NV = 8.0*D_v*rho[0]*rho[1]*np.array([[1., -1.], [-1., 1.]])
    return dict(theta=theta, gamma=gamma, A=A, Q=Q, NS=NS, NV=NV)


def spatial_symbols(N=(128, 128), L=(1., 1.), symbol='fv2'):
    """q² for FV face differences, and -Laplacian symbol for curvature.

    fv2:      nearest-neighbor face differences and second-order Laplacian.
    continuum: exact k² for both operators (long-wavelength SI theory).
    Requires periodic boundaries; these symbols do not describe Neumann BC.
    """
    nx, ny = map(int, N)
    dx, dy = float(L[0])/nx, float(L[1])/ny
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dy)
    x, y = np.meshgrid(kx, ky, indexing='ij')
    k2 = x*x + y*y
    if symbol == 'continuum':
        return x, y, k2, k2
    face = 4*np.sin(0.5*x*dx)**2/dx**2 + 4*np.sin(0.5*y*dy)**2/dy**2
    if symbol == 'fv2':
        lap = face
    else:
        raise ValueError("symbol must be 'fv2' or 'continuum'")
    return x, y, face, lap


def check_periodic_solver_laplacian(solver, modes=((1, 0), (0, 1), (3, 4))):
    """Check the solver's actual D2x, D2y against the fv2 Fourier symbol.

    For periodic, fft=False simulations. Returns a dictionary of mode ->
    maximum absolute eigenvector residual, i.e. it is near zero when the
    installed spatial operators match the theory exactly.
    """
    if solver.bc != 'periodic' or solver.fft:
        raise ValueError('Use bc="periodic" and fft=False for this test')
    nx, ny = solver.Nx, solver.Ny
    x = np.arange(nx)
    y = np.arange(ny)
    result = {}
    for mx, my in modes:
        mx, my = int(mx) % nx, int(my) % ny
        wave = np.exp(2j*np.pi*mx*x/nx)[:, None] * np.exp(2j*np.pi*my*y/ny)[None, :]
        numerical = solver.D2x.dot(wave) + solver.D2y.dot(wave.T).T
        q2 = 4*np.sin(np.pi*mx/nx)**2/solver.dx**2 + 4*np.sin(np.pi*my/ny)**2/solver.dy**2
        result[(mx, my)] = float(np.max(np.abs(numerical + q2*wave)))
    return result


def finite_time_covariance(Lmat, Nmat, time):
    """Compute ∫_0^t e^(L s) N e^(L.T s) ds, stable OR unstable.

    Uses a 2x2 eigen-decomposition. Fall back to a 5x5 block exponential
    when the eigenbasis is poorly conditioned (exceptional points).
    """
    if time < 0:
        raise ValueError('time must be >= 0')
    if time == 0:
        return np.zeros((2, 2))
    ev, V = np.linalg.eig(Lmat)
    if np.linalg.cond(V) > 1e8:
        from scipy.linalg import expm
        K = np.kron(np.eye(2), Lmat) + np.kron(Lmat, np.eye(2))
        mat = np.zeros((5, 5)); mat[:4, :4] = K; mat[:4, 4] = Nmat.reshape(-1, order='F')
        C = expm(mat*time)[:4, 4].reshape(2, 2, order='F')
        return 0.5*(C+C.T)
    Vi = np.linalg.inv(V)
    B = Vi @ Nmat @ Vi.T
    lam = ev[:, None] + ev[None, :]
    f = np.empty_like(lam, dtype=complex)
    small = np.abs(lam*time) < 1e-8
    f[small] = time*(1 + 0.5*lam[small]*time + (lam[small]*time)**2/6)
    f[~small] = np.expm1(lam[~small]*time)/lam[~small]
    C = (V @ (B*f) @ V.T).real
    return 0.5*(C+C.T)


def spectrum(N=(128, 128), L=(1., 1.), rho=(0.35, 0.35), D=(1., 1.),
             beta=5., kappa=((0.6, -0.4), (-0.4, 0.6)), h=0.001,
             D_v=0., symbol='fv2', time=None, initial_cell_std=(0., 0.)):
    """Return spectrum arrays S[2,2,Nx,Ny], growth rates and stability mask.

    time=None: stationary covariance only for stable modes (unstable = NaN).
    time>=0: covariance of zero-initial fluctuations at finite time, even if
    deterministic modes are unstable. To include nonzero initial fluctuations,
    add exp(Lt) C_initial exp(L.T t), where initial_cell_std gives the
    independent per-cell initial standard deviations of A and B.

    FFT comparison: S_sim[k] = (Lx*Ly/(Nx*Ny)**2) * FFT(delta_rho)[k]
                         * conj(FFT(delta_rho)[k]), with delta_rho=phi-mean.
    This is |integral d²x e^{-ikx} delta_rho|² / (Lx*Ly).
    No radial averaging is done here: the full two-dimensional matrix spectrum
    is returned for anisotropic finite-difference grids.
    """
    mats = linear_matrices(rho, D, beta, kappa, h, D_v)
    kx, ky, q2, lap = spatial_symbols(N, L, symbol)
    S = np.full((2, 2, *q2.shape), np.nan)
    std0 = np.broadcast_to(np.asarray(initial_cell_std, dtype=float), (2,))
    if np.any(std0 < 0) or (time is None and np.any(std0)):
        raise ValueError('initial_cell_std must be >=0, and needs finite --time')
    C0 = np.diag(std0**2) * (float(L[0])*float(L[1]) / (int(N[0])*int(N[1])))
    growth = np.empty((*q2.shape, 2)); stable = np.zeros(q2.shape, dtype=bool)
    for ij in np.ndindex(q2.shape):
        u, v = q2[ij], lap[ij]
        linear = -u*mats['A'] + u*v*mats['Q']
        noise = mats['NV'] + u*mats['NS']
        eig = np.linalg.eigvals(linear)
        growth[ij] = np.sort(eig.real)
        is_stable = (u > 1e-14 and eig.real.max() < -1e-12)
        stable[ij] = is_stable
        if time is not None:
            C = finite_time_covariance(linear, noise, time)
            if np.any(std0):
                E = expm(linear*time)
                C += E @ C0 @ E.T
        elif is_stable:
            C = solve_continuous_lyapunov(linear, -noise)
            C = 0.5*(C+C.T)
        else:
            continue
        S[:, :, ij[0], ij[1]] = C
    return dict(S=S, kx=kx, ky=ky, q2=q2, lap_q2=lap, growth=growth,
                stable=stable, **mats)


def radial_average(output, nbins=60, field='S', component=(0, 0), stable_only=True):
    """Azimuthal average of one spectral component; excludes k=0."""
    values = output[field][component] if field == 'S' else output[field]
    k = np.hypot(output['kx'], output['ky'])
    mask = np.isfinite(values) & (k > 0)
    if stable_only:
        mask &= output['stable']
    edges = np.linspace(0.0, np.max(k), nbins+1)
    ib = np.digitize(k[mask], edges) - 1
    count = np.bincount(ib, minlength=nbins)[:nbins]
    sums = np.bincount(ib, weights=values[mask], minlength=nbins)[:nbins]
    return 0.5*(edges[1:]+edges[:-1]), np.divide(sums, count, out=np.full(nbins, np.nan), where=count>0), count


def empirical_fft_spectrum(phi_samples, L=(1., 1.)):
    """Ensemble structure factor in the same normalization as spectrum().

    phi_samples shape (samples,2,Nx,Ny); remove spatial mean PER sample.
    Returns a real symmetric 2x2 covariance spectrum (equal-time co-spectrum).
    """
    arr = np.asarray(phi_samples, dtype=float)
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError('samples must have shape (samples,2,Nx,Ny)')
    centered = arr - arr.mean(axis=(-2, -1), keepdims=True)
    fk = np.fft.fft2(centered, axes=(-2, -1))
    factor = (L[0]*L[1])/(arr.shape[2]*arr.shape[3])**2
    G = factor * np.einsum('saxy,sbxy->abxy', fk, fk.conj()).real / arr.shape[0]
    return 0.5*(G + G.swapaxes(0, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, nargs=2, default=(128, 128)); p.add_argument('--L', type=float, nargs=2, default=(1, 1))
    p.add_argument('--rho', type=float, nargs=2, default=(.35, .35)); p.add_argument('--D', type=float, nargs=2, default=(1, 1))
    p.add_argument('--beta', type=float, default=0.5); p.add_argument('--h', type=float, default=.001)
    p.add_argument('--kappa', type=float, nargs=4, default=(.6, -.4, -.4, .6))
    p.add_argument('--D-v', type=float, default=0.0)
    p.add_argument('--symbol', choices=('fv2', 'continuum'), default='fv2')
    p.add_argument('--time', type=float, default=None, help='Finite-time covariance; omit for stationary stable modes')
    p.add_argument('--initial-cell-std', type=float, nargs=2, default=(0.,0.),
                   help='Initial per-cell A,B standard deviations, independent between cells; requires --time')
    p.add_argument('--output', default='schelling_spectrum_fv2.npz')
    p.add_argument('--plot', default=None, help='Optional PNG path for the radial AA, AB, BB spectra')
    a = p.parse_args()
    r = spectrum(a.N, a.L, a.rho, a.D, a.beta, np.array(a.kappa).reshape(2, 2),
                 a.h, a.D_v, a.symbol, a.time, a.initial_cell_std)
    np.savez_compressed(a.output, **r)
    print('Saved:', a.output)
    print('Theta=\n', r['theta'], '\nGamma=\n', r['gamma'])
    print('A=\n', r['A'], '\nQ=\n', r['Q'])
    print('Stable nonzero modes:', int(r['stable'].sum()), '/', np.prod(a.N)-1)
    print('Maximum real growth rate:', np.max(r['growth'][..., -1]))
    if a.time is None and np.count_nonzero(r['stable']) == 0:
        print('No stationary spectrum: all nonzero modes are unstable. Try --time 1e-5.')
    if a.plot:
        import matplotlib.pyplot as plt
        for component, label in [((0,0),'AA'),((0,1),'AB'),((1,1),'BB')]:
            k, s, n = radial_average(r, field='S', component=component, stable_only=a.time is None)
            plt.plot(k, s, label=label)
        plt.xlabel(r'$|k|$'); plt.ylabel(r'$S_{ab}(k)$ [area units]')
        plt.legend(); plt.tight_layout(); plt.savefig(a.plot, dpi=180)
        print('Saved:', a.plot)


if __name__ == '__main__':
    main()
