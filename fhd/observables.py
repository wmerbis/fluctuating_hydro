"""Spatial and spatiotemporal observables for two-species Schelling models.

Works directly with FHD ``phi_run`` of shape (2, T, Nx, Ny), or with one
configuration (2, Nx, Ny).  Physical domain defaults to L=(50,50).The same 
density-field interface works for a coarse-grained ABM (use counts divided 
by the number of lattice sites/box).

Units and conventions
---------------------
* The spatial transform is unitary. S_ab(k) = dx*dy * F_a(k) F_b(k)^*,
  with F=fft2(delta_rho, norm='ortho') for periodic BC; DCT-II with
  norm='ortho' for cell-centered Neumann BC.
* k=0 is excluded from all spectral statistics (conserved global masses).
* FFT k vectors are 2*pi*fftfreq(N, dx); Neumann DCT k=pi*n/L.
* k_definition='physical' bins those continuum mode labels (legacy default).
  k_definition='lattice' bins the exact second-order FV Laplacian
  eigen-wavenumber: 2*sin(kx*dx/2)/dx in each direction. Using lattice k
  changes BIN MEMBERSHIP, not just x-axis labels.

* Dynamic structure factor S_ab(k,omega) uses angular frequencies and
  satisfies int S(k,omega) d omega/(2*pi) = time-averaged equal-time
  connected S(k), for a rectangular time window.
* Excess reference 'initial' subtracts the first frame's MODE-WISE power;
  'white' subtracts a wavevector-independent reference estimated from the
  initial covariance; 'beta0' is the canonical, fixed-mass ideal-exclusion
  reference (requiring an appropriate effective cell capacity). An actual
  beta=0 reference trajectory can also be provided as an array.
* Radial averaging removes directional information. For coherent migration,
  inspect individual k modes as well as the isotropic radial spectrum.

Only NumPy and SciPy (for the DCT) are required. No solver imports.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import dctn, fft as time_fft, ifft as time_ifft, next_fast_len

__all__ = [
    'spatial_structure_factor', 'excess_spectral_power',
    'beta0_reference_covariance',
    'segregation_index', 'occupation_index',
    'connected_time_lag_covariance', 'spatiotemporal_structure_factor',
    'estimate_linear_eigenvalues', 'eigenvalues_from_covariance',
    'compute_observables', 'radial_wavenumber_map',
    'radial_bin_spectrum_2d',
]


def _frames(phi_run):
    a = np.asarray(phi_run, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 2:
        a = a[:, None, :, :]
    if a.ndim != 4 or a.shape[0] != 2:
        raise ValueError('phi_run must have shape (2,T,Nx,Ny) or (2,Nx,Ny)')
    if min(a.shape[1:]) < 1 or not np.all(np.isfinite(a)):
        raise ValueError('Empty dimensions or non-finite densities')
    return a


def _geometry(shape, L, bc, n_bins, bin_edges, k_max, k_definition='physical'):
    nx, ny = shape
    lx, ly = map(float, L)
    if lx <= 0 or ly <= 0:
        raise ValueError('L must be positive')
    area_cell = lx * ly / (nx * ny)
    bc = bc.lower()
    if bc in ('periodic', 'fft'):
        kx = 2*np.pi*np.fft.fftfreq(nx, d=lx/nx)
        ky = 2*np.pi*np.fft.fftfreq(ny, d=ly/ny)
        transform = 'fft'
    elif bc in ('neumann', 'dct'):
        kx = np.pi*np.arange(nx)/lx
        ky = np.pi*np.arange(ny)/ly
        transform = 'dct'
    else:
        raise ValueError("bc must be 'periodic' or 'Neumann'")
    # Physical (continuum) mode labels and lattice-effective wavenumbers.
    # The second-order finite-volume Laplacian acts on each mode as
    #    D2_mode = -k_lattice**2 * mode.
    # Neumann uses DCT-II modes n*pi/L; periodic uses FFT modes 2*pi*n/L.
    k_physical = np.hypot(kx[:, None], ky[None, :])
    k_lattice = np.hypot(
        2.0*np.sin(0.5*kx[:, None]*(lx/nx))/(lx/nx),
        2.0*np.sin(0.5*ky[None, :]*(ly/ny))/(ly/ny))
    if k_definition not in ('physical', 'lattice'):
        raise ValueError("k_definition must be 'physical' or 'lattice'")
    k = k_lattice if k_definition == 'lattice' else k_physical
    if bin_edges is None:
        # Grid wavenumber range, inferred from L, N and BC. Users
        # should normally leave k_max=None, particularly when changing L.
        k_grid_max = float(np.max(k))
        if k_max is None:
            k_hi = k_grid_max
        else:
            k_hi = float(k_max)
            if not np.isfinite(k_hi) or k_hi <= 0:
                raise ValueError('k_max must be positive')
            if k_hi > k_grid_max:
                warnings.warn(
                    f'k_max={k_hi:g} exceeds the largest grid wavenumber '
                    f'{k_grid_max:.6g} for N={shape}, L={L}, bc={bc}; '
                    'using the grid maximum instead. Pass k_max=None to '
                    'select it automatically.', UserWarning, stacklevel=2)
                k_hi = k_grid_max
        if n_bins < 1:
            raise ValueError('n_bins must be >= 1')
        edges = np.linspace(0.0, k_hi * (1+1e-12), int(n_bins)+1)
    else:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges)>0):
            raise ValueError('bin_edges must be strictly increasing')
    which = np.searchsorted(edges, k.ravel(), side='right')-1
    # Never include the exactly-zero mode, irrespective of bin edge settings.
    valid = (k.ravel()>0) & (which>=0) & (which<len(edges)-1)
    count = np.bincount(which[valid], minlength=len(edges)-1)
    b = which[valid]
    def mode_mean(values):
        out = np.full(len(count), np.nan)
        np.divide(np.bincount(b, weights=values.ravel()[valid],
                              minlength=len(count)), count,
                  out=out, where=count > 0)
        return out
    kp_mean = mode_mean(k_physical)
    kl_mean = mode_mean(k_lattice)
    kl_var = mode_mean(k_lattice**2) - kl_mean**2
    return dict(k=k, k_physical=k_physical, k_lattice=k_lattice,
                k_definition=k_definition,
                k_mean_physical=kp_mean, k_mean_lattice=kl_mean,
                k_std_lattice=np.sqrt(np.maximum(kl_var, 0)),
                edges=edges, centers=(edges[:-1]+edges[1:])/2,
                valid=valid, selected=np.flatnonzero(valid),
                bins=b, counts=count, area_cell=area_cell,
                transform=transform)


def radial_wavenumber_map(N, L=(50., 50.), *, bc='Neumann',
                          n_bins=40, bin_edges=None, k_max=None,
                          k_definition='physical'):
    """Map radial-bin *membership* between physical and FV lattice k.

    With k_definition='physical', passing the saved ``k_edges`` from an
    older sweep recovers exactly its original modes per bin. The returned
    k_lattice_mean is an APPROXIMATE axis conversion only: the old bin
    powers cannot be reconstructed as exact lattice-k bins without the
    original DCT coefficients (i.e. the density snapshots/trajectory).
    """
    g = _geometry(tuple(map(int,N)), L, bc, n_bins, bin_edges, k_max,
                  k_definition)
    return dict(k=g['centers'], k_edges=g['edges'], counts=g['counts'],
                k_definition=k_definition,
                k_physical_mean=g['k_mean_physical'],
                k_lattice_mean=g['k_mean_lattice'],
                k_lattice_std=g['k_std_lattice'])


def radial_bin_spectrum_2d(S2d, L=(50., 50.), *, bc='Neumann',
                           n_bins=40, bin_edges=None, k_max=None,
                           k_definition='lattice'):
    """Azimuthally average an existing 2D theoretical/mode-resolved spectrum.

    S2d shape (..., Nx, Ny), e.g. (2,2,Nx,Ny). The last two axes
    must be on the same DCT/FFT mode grid as the simulation. With
    bin_edges=simulation['k_edges'] and the same k_definition, this
    reproduces EXACTLY the simulation's bin membership and normalization
    (it averages values, but does not multiply by any FFT/DCT scaling).
    """
    a = np.asarray(S2d)
    if a.ndim < 2:
        raise ValueError('S2d needs spatial mode axes (Nx, Ny)')
    g = _geometry(a.shape[-2:], L, bc, n_bins, bin_edges, k_max,
                  k_definition)
    nbin = len(g['counts'])
    flat = a.reshape(-1, a.shape[-2]*a.shape[-1])[:,g['selected']]
    avg = np.empty((flat.shape[0], nbin), dtype=complex)
    for j in range(flat.shape[0]):
        avg[j] = _bin_complex(flat[j], g['bins'], g['counts'], nbin)
    avg = avg.reshape(a.shape[:-2]+(nbin,))
    if np.isrealobj(a):
        avg = avg.real
    return dict(k=g['centers'], k_edges=g['edges'], counts=g['counts'],
                S=avg, **_k_metadata(g))


def _k_metadata(geo):
    return dict(k_definition=geo['k_definition'],
                k_physical_mean=geo['k_mean_physical'],
                k_lattice_mean=geo['k_mean_lattice'],
                k_lattice_std=geo['k_std_lattice'])


def _transform(phi, transform):
    # Spatial demeaning fixes the mass-conserving zero mode exactly.
    u = phi - phi.mean(axis=(-2,-1), keepdims=True)
    if transform == 'fft':
        return np.fft.fft2(u, axes=(-2,-1), norm='ortho')
    return dctn(u, axes=(-2,-1), type=2, norm='ortho')


def _modes(phi, geo, start=0, stop=None):
    # Output (species, time, retained modes). Most moderate-k analyses
    # retain far fewer than Nx*Ny modes, reducing dynamic-analysis RAM.
    arr = _frames(phi)
    idx = geo['selected']
    if stop is None: stop=arr.shape[1]
    transformed = _transform(arr[:, start:stop], geo['transform'])
    return transformed.reshape(2, stop-start, -1)[..., idx]


def _bin_complex(vals, bins, count, n):
    """Radial mean for possibly complex mode weights, with zero empty bins."""
    re = np.bincount(bins, weights=np.real(vals), minlength=n)
    im = np.bincount(bins, weights=np.imag(vals), minlength=n)
    z = (re + 1j*im) / np.maximum(count, 1)
    z[count==0] = np.nan+1j*np.nan
    return z


def _bin_covariances(modes, geo):
    # Input (2, T, modes); output (T, bins, 2, 2).
    nt = modes.shape[1]
    nb = len(geo['counts'])
    ret = np.empty((nt, nb, 2, 2), dtype=np.complex128)
    for t in range(nt):
        for a in range(2):
            for b in range(2):
                ret[t, :, a, b] = geo['area_cell'] * _bin_complex(
                    modes[a,t]*modes[b,t].conj(), geo['bins'], geo['counts'], nb)
    return ret


def _components(S):
    # [...,2,2], the equal-species normalized +/- combinations
    s_ab = np.real(S[...,0,1]+S[...,1,0])
    return np.real(0.5*(S[...,0,0]+S[...,1,1]+s_ab)), np.real(
        0.5*(S[...,0,0]+S[...,1,1]-s_ab))


def beta0_reference_covariance(rho, *, cell_capacity=1., n_cells=None):
    """Nonzero-mode covariance of a uniform, fixed-mass A/B/vacancy gas.

    Each coarse cell averages ``cell_capacity`` independent exclusion sites.
    For a microscopic one-site-per-cell lattice set cell_capacity=1.
    The canonical finite-size correction is K/(K-1), K=n_cells*capacity;
    the grand-canonical (large-system) limit is simply Sigma/capacity.
    Do not use this amplitude uncalibrated for an SPDE with arbitrary noise.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (2,) or not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError('rho must contain two nonnegative finite densities')
    if rho.sum() > 1 + 1.e-10:
        raise ValueError('rho_A + rho_B must not exceed 1')
    cap = float(cell_capacity)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('cell_capacity must be finite and positive')
    cov = (np.diag(rho) - np.outer(rho, rho)) / cap
    if n_cells is not None:
        n_total = float(n_cells)*cap
        if n_total <= 1:
            raise ValueError('At least two exclusion sites required')
        cov *= n_total/(n_total-1.)
    return cov


def _mode_baseline(phi, geo, reference, *, reference_cell_capacity=1.,
                   reference_rho=None):
    """One 2x2 baseline spectrum per radial bin (area_cell normalization)."""
    nb = len(geo['counts'])
    if isinstance(reference, str):
        if reference not in ('initial', 'white', 'none', 'beta0'):
            raise ValueError("reference must be 'initial', 'white', 'beta0', 'none', or an array")
        if reference == 'none':
            return np.zeros((nb,2,2), dtype=complex)
        if reference == 'beta0':
            rho = (phi[:,0].mean(axis=(-2,-1)) if reference_rho is None
                   else reference_rho)
            cov = beta0_reference_covariance(
                rho, cell_capacity=reference_cell_capacity,
                n_cells=phi.shape[-2]*phi.shape[-1])
            baseline = geo['area_cell']*cov
            return np.broadcast_to(baseline,(nb,2,2)).copy().astype(complex)
        ref_phi = phi[:,0:1]
    else:
        ref_phi = _frames(reference)
        if ref_phi.shape[-2:] != phi.shape[-2:]:
            raise ValueError('reference field must have the same spatial grid')
    if isinstance(reference,str) and reference=='white':
        # White, estimated from the actual initial spatial covariance.
        z = ref_phi[:,0].reshape(2,-1)
        cc = np.cov(z, ddof=1)
        baseline = geo['area_cell'] * cc
        return np.broadcast_to(baseline,(nb,2,2)).copy().astype(complex)    
    modes = _modes(ref_phi,geo)
    with np.errstate(invalid='ignore'):
        return np.mean(_bin_covariances(modes,geo),axis=0)


def spatial_structure_factor(phi_run, L=(50.,50.), *, bc='periodic',
                             n_bins=40, bin_edges=None, k_max=None,
                             k_definition='physical'):
    """Radially averaged equal-time S_ab(k,t) including the A/B cross spectrum.

    Returns dict with `S` of shape (T,n_bins,2,2), `S_plus`, `S_minus`
    (T,n_bins), `k`, `k_edges`, `counts`. If a bin has no modes its power
    is NaN. k=0 is always excluded. Callers can select a low-k cutoff.
    """
    phi = _frames(phi_run)
    geo = _geometry(phi.shape[-2:],L,bc,n_bins,bin_edges,k_max,k_definition)
    S = _bin_covariances(_modes(phi,geo),geo)
    sp, sm = _components(S)
    return dict(k=geo['centers'], k_edges=geo['edges'], counts=geo['counts'],
                S=S, S_plus=sp, S_minus=sm, bc=bc,
                **_k_metadata(geo))


def excess_spectral_power(phi_run, L=(50.,50.), *, reference='beta0',
                          bc='periodic', n_bins=40, bin_edges=None,
                          k_max=None, k_band=None, k_definition='physical',
                          threshold=1e-13,
                          min_excess_fraction=0.10,
                          reference_cell_capacity=None, reference_rho=None, h = None):
    """Excess spectra and segregation/integration order parameter C(t).

    reference='initial' uses actual first frame's spectrum; 'white' uses
    spatial covariance of the initial field, giving a flat baseline;
    'beta0' uses the exact ideal-exclusion fixed-mass covariance, with
    cell_capacity independent exclusion sites per coarse cell. For stochastic
    density PDEs, this microscopic amplitude may NOT match the solver noise:
    an equilibrated beta=0 trajectory supplied as an array is safer.
    'none' uses zero. Reference arrays have shape (2,Nx,Ny) or
    (2,Tref,Nx,Ny) and represent an ensemble of reference configurations.

    Integrated excess P_{+,-} sums [S_{+,-}(t)-S_{+,-}^ref] over EVERY
    Fourier mode in chosen k-band, not equally over radial bins. P can be
    signed. To get robust C, P_{+,-} are individually clamped to >=0, and
    C=(Pplus_pos-Pminus_pos)/(Pplus_pos+Pminus_pos). If both are tiny,
    C=NaN (a homogeneous state has no well-defined phase-channel label).
    Values of C are masked unless combined positive excess exceeds
    max(threshold,min_excess_fraction*(baseline plus+minus power)).
    This is a pattern CHANNEL diagnostic, not an independent Z2 order param.
    """
    phi = _frames(phi_run)
    # For the analytical beta=0 reference, infer the effective
    # number of microscopic exclusion sites per numerical cell.
    if isinstance(reference, str) and reference == "beta0":
        if reference_cell_capacity is None:
            if h is None or not np.isfinite(h) or h <= 0:
                raise ValueError(
                    "reference='beta0' requires h or "
                    "an explicit reference_cell_capacity"
                )

            nx, ny = phi.shape[-2:]
            dx = float(L[0]) / nx
            dy = float(L[1]) / ny
            reference_cell_capacity = dx * dy / h**2
    geo = _geometry(phi.shape[-2:],L,bc,n_bins,bin_edges,k_max,k_definition)
    S = _bin_covariances(_modes(phi,geo),geo)
    baseline = _mode_baseline(
        phi,geo,reference, reference_cell_capacity=reference_cell_capacity,
        reference_rho=reference_rho)
    excess = S - baseline[None,:,:,:]
    plus,minus = _components(S)
    plus0,minus0 = _components(baseline)
    eplus,eminus = plus-plus0[None], minus-minus0[None]
    use = geo['counts']>0
    if k_band is not None:
        lo,hi = map(float,k_band)
        use &= (geo['centers']>=lo)&(geo['centers']<=hi)
    if not np.any(use):
        raise ValueError('No Fourier modes lie within the selected k_band')
    weight = geo['counts'][use]
    Pplus = np.nansum(eplus[:,use]*weight[None],axis=1)
    Pminus = np.nansum(eminus[:,use]*weight[None],axis=1)
    pp=np.maximum(Pplus,0.); pm=np.maximum(Pminus,0.)
    den=pp+pm
    # No sensible phase-channel label if the excess structure is smaller than
    # a selected fraction of the initial/reference power in that k band.
    # Increase this fraction for conservative phase maps near the transition.
    bplus,bminus=_components(baseline)
    ref_power = float(np.nansum((bplus[use]+bminus[use])*weight))
    cutoff = max(float(threshold),float(min_excess_fraction)*max(0.,ref_power))
    C=np.full(len(Pplus), np.nan)
    np.divide(pp-pm, den, out=C, where=den>cutoff)
    return dict(k=geo['centers'],k_edges=geo['edges'], counts=geo['counts'],
                S=S, baseline=baseline, excess=excess,
                S_plus=plus,S_minus=minus,excess_plus=eplus,
                excess_minus=eminus,P_plus=Pplus,P_minus=Pminus,C=C,
                k_band=k_band, order_threshold=cutoff,
                reference_total_power=ref_power, **_k_metadata(geo))


def segregation_index(phi_run, *, occupancy_threshold=1e-12,
                      cell_capacity=None, subtract_mixing=False):
    """Occupancy-weighted composition variance per frame.

    I_seg = sum_i n_i [rhoA_i/n_i - rhoA_global/n_global]^2 / sum_i n_i.
    Empty cells excluded. With cell_capacity=N_sites_per_cell, return the
    approximate random-mixing shot-noise baseline p(1-p) * N_nonempty /
    N_particles (N_particles=cell_capacity*sum n_i). This baseline assumes
    random binomial species labeling conditional on occupancies; it is only
    approximate when global totals are fixed (hypergeometric correction).
    `excess` is raw unless subtract_mixing=True AND cell_capacity supplied.
    """
    a=_frames(phi_run)
    n=a.sum(axis=0)
    total=n.sum(axis=(-2,-1))
    glob_a=a[0].sum(axis=(-2,-1))/np.maximum(total,1e-300)
    c=np.divide(a[0], n, out=np.zeros_like(n), where=n>occupancy_threshold)
    numerator=np.sum(n*(c-glob_a[:,None,None])**2,axis=(-2,-1),
                     where=n>occupancy_threshold)
    raw=numerator/np.maximum(total,1e-300)
    mixing=None
    if cell_capacity is not None:
        if cell_capacity<=0: raise ValueError('cell_capacity must be positive')
        n_boxes=np.count_nonzero(n>occupancy_threshold,axis=(-2,-1))
        mixing=glob_a*(1-glob_a)*n_boxes / (cell_capacity*np.maximum(total,1e-300))
    if subtract_mixing and mixing is None:
        raise ValueError('Provide cell_capacity to subtract mixing baseline')
    return dict(I_seg=raw, mixing_baseline=mixing,
                excess=raw if mixing is None or not subtract_mixing else raw-mixing,
                global_A_fraction=glob_a)


def occupation_index(phi_run, *, reference='initial'):
    """Spatial variance of n=rhoA+rhoB at each frame; initial excess too.

    Reference can be 'initial' (initial variance), 'none' (0), or another
    density field / ensemble, averaged over reference frames.
    """
    a=_frames(phi_run)
    n=a.sum(axis=0)
    var=np.var(n,axis=(-2,-1))
    if isinstance(reference,str):
        if reference=='initial': base=float(var[0])
        elif reference=='none': base=0.
        else: raise ValueError("reference must be 'initial', 'none', or an array")
    else:
        b=_frames(reference)
        base=float(np.var(b.sum(axis=0),axis=(-2,-1)).mean())
    return dict(I_occ=var, baseline=base, excess=var-base)


def _validated_lags(lags, available_frames):
    """Choose only feasible lags after burn-in, reporting discarded requests."""
    if lags is None:
        return np.arange(min(100, available_frames - 1) + 1, dtype=int)
    if isinstance(lags, str):
        if lags != 'all':
            raise ValueError("lags must be None, 'all', or integer frame offsets")
        return np.arange(available_frames, dtype=int)
    raw = np.asarray(lags)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError('lags must be a nonempty one-dimensional array')
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError('lags must contain finite integer frame offsets')
    if np.any(raw != np.floor(raw)) or np.any(raw < 0):
        raise ValueError('lags must be nonnegative integer frame offsets')
    requested = raw.astype(int)
    keep = requested < available_frames
    if not np.any(keep):
        raise ValueError(
            f'No valid lags: only {available_frames} frames remain after '
            f'burn_in, so requested lags must be in [0, {available_frames - 1}]. '
            'Pass lags=None for automatic selection.')
    if not np.all(keep):
        warnings.warn(
            f'Discarded {np.count_nonzero(~keep)} requested lag(s) >= '
            f'{available_frames}; only {available_frames} frames remain '
            'after burn_in. Pass lags=None for automatic selection.',
            UserWarning, stacklevel=3)
    return requested[keep]


def connected_time_lag_covariance(phi_run, dt, L=(50.,50.), *,
                                  bc='periodic', n_bins=40, bin_edges=None,
                                  k_max=None, lags=None, burn_in=0,
                                  k_definition='physical',
                                  demean_time=False, method='fft',
                                  mode_chunk_size=256):
    """Lag covariances of orthonormal spatial modes, averaged over radial bins.

    C_ab(k,tau) = area_cell <z_a(t+tau,k) z_b(t,k)^*>_{t,k in bin}.
    ``lags='all'`` computes every lag after burn-in; ``lags=None`` retains
    the old 0..100 default. Spatial means are always removed, but temporal
    demeaning is off by default to avoid suppressing growing slow modes. ``method='fft'`` computes the EXACT same finite
    sample/unbiased-lag estimator using zero-padded FFT convolutions, with
    O(T log T) time scaling. Modes are processed in bounded chunks.

    Note: the longest lags have very few time pairs, and should not be used
    indiscriminately in quantitative phase classification.
    """
    a = _frames(phi_run)
    if dt<=0: raise ValueError('dt must be the spacing BETWEEN SAVED frames')
    if burn_in<0 or burn_in>=a.shape[1]-1:
        raise ValueError('burn_in must leave >=2 frames')
    geo=_geometry(a.shape[-2:],L,bc,n_bins,bin_edges,k_max,k_definition)
    z=_modes(a,geo,start=burn_in)
    T=z.shape[1]
    if demean_time: z=z-z.mean(axis=1,keepdims=True)
    lags = _validated_lags(lags, T)
    nb=len(geo['counts'])
    cov=np.full((len(lags),nb,2,2), np.nan+1j*np.nan,dtype=complex)
    if method == 'direct':
        for li, lag in enumerate(lags):
            for species in range(2):
                for other in range(2):
                    per_mode=np.einsum('tm,tm->m', z[species,lag:],
                                      z[other,:T-lag].conj(),
                                      optimize=True)/(T-lag)
                    cov[li,:,species,other] = geo['area_cell']*_bin_complex(
                        per_mode,geo['bins'],geo['counts'],nb)
    elif method == 'fft':
        if mode_chunk_size < 1:
            raise ValueError('mode_chunk_size must be positive')
        # Padding by >=T-1 zeros eliminates circular wraparound: at lag l
        # ifft(fft(z_a)*conj(fft(z_b)))[l] = sum_t z_a(t+l)*conj(z_b(t)).
        nfft = next_fast_len(2*T-1)
        cross_spectra=np.zeros((nb,2,2,nfft),dtype=complex)
        nm=z.shape[-1]
        for left in range(0,nm,mode_chunk_size):
            right=min(nm,left+mode_chunk_size)
            freq=time_fft(z[:,:,left:right],n=nfft,axis=1)
            local_bins=geo['bins'][left:right]
            for ib in np.unique(local_bins):
                selected=local_bins==ib
                sub=freq[:,:,selected]
                for species in range(2):
                    for other in range(2):
                        cross_spectra[ib,species,other] += np.sum(
                            sub[species]*sub[other].conj(),axis=-1)
        for ib in range(nb):
            if geo['counts'][ib]==0:
                continue
            # numpy/scipy inverse FFT uses normalization by 1/nfft.
            corr=time_ifft(cross_spectra[ib],axis=-1)
            corr=corr[...,lags]/(T-lags)[None,None,:]
            cov[:,ib]=np.moveaxis(corr,-1,0)*geo['area_cell']/geo['counts'][ib]
    else:
        raise ValueError("method must be 'fft' or 'direct'")
    plus,minus=_components(cov)
    return dict(lags=lags,tau=lags*float(dt),k=geo['centers'],
                counts=geo['counts'],C=cov,C_plus=plus,C_minus=minus,
                demean_time=demean_time,method=method,
                n_frames_used=T,dt_saved=float(dt))


def spatiotemporal_structure_factor(phi_run, dt, L=(50.,50.), *,
                                     bc='periodic', n_bins=40,
                                     bin_edges=None,k_max=None,burn_in=0,
                                     k_definition='physical',
                                     window='hann',demean_time=False,
                                     return_mode_spectrum=False):
    """S_ab(k-bin, omega), with angular omega in rad / physical time.

    S_ab(k,omega) = area_cell*dt/sum(w_t**2) *
       < FFT_t[w_t F_a(k,t)] FFT_t[w_t F_b(k,t)]^* >_k.
    Output S has shape (n_bins,n_omega,2,2). Two-sided frequencies including
    negative omega. For `window='boxcar'`, integral S d omega/(2*pi)
    equals the time-mean connected spatial structure factor exactly.

    return_mode_spectrum=True returns full mode-resolved data and kx,ky
    indices, useful for identifying directional propagation; may be large.
    """
    a=_frames(phi_run)
    if dt<=0: raise ValueError('dt must be the spacing BETWEEN SAVED frames')
    if burn_in<0 or burn_in>=a.shape[1]-1:
        raise ValueError('burn_in must leave >=2 frames')
    geo=_geometry(a.shape[-2:],L,bc,n_bins,bin_edges,k_max,k_definition)
    z=_modes(a,geo,start=burn_in)
    T=z.shape[1]
    if demean_time: z=z-z.mean(axis=1,keepdims=True)
    if window in ('boxcar','rect','none',None): w=np.ones(T)
    elif window=='hann': w=np.hanning(T)
    else: raise ValueError("window must be 'hann' or 'boxcar'")
    if not np.any(w): raise ValueError('Time window vanishes: use >=3 samples for hann')
    W=np.fft.fft(z*w[None,:,None],axis=1)
    scale=geo['area_cell']*float(dt)/np.sum(w*w)
    nb=len(geo['counts'])
    S=np.full((nb,T,2,2),np.nan+0j,dtype=complex)
    mode_spectrum={} if return_mode_spectrum else None
    for ib in range(nb):
        keep=(geo['bins']==ib)
        if not np.any(keep): continue
        for aa in range(2):
            for bb in range(2):
                f=scale*W[aa,:,keep]*W[bb,:,keep].conj()
                S[ib,:,aa,bb]=np.mean(f,axis=0)
    omega=2*np.pi*np.fft.fftfreq(T,d=dt)
    sort=np.argsort(omega)
    Sp,Sm=_components(S)
    ret=dict(k=geo['centers'], k_edges=geo['edges'],counts=geo['counts'],
             omega=omega[sort], frequency=omega[sort]/(2*np.pi),
             S=S[:,sort],S_plus=Sp[:,sort],S_minus=Sm[:,sort],
             window=window,dt_saved=dt, **_k_metadata(geo))
    if return_mode_spectrum:
        # The phases and cross spectra of specific directional k vectors
        # can be analyzed from F; do not use its circularly averaged power
        # to infer a net direction of motion.
        ret['mode_temporal_fft']=W[:,sort]
        ret['mode_flat_indices']=geo['selected']
        ret['k_grid']=geo['k']
        ret['mode_spectrum_scale']=scale
    return ret

def _scanning_bands(k_range, widths, step):
    """Physical-k windows, including the rightmost window of every width."""
    lo, hi = map(float, k_range)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError('k_range must satisfy 0 < lower < upper')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be positive')
    bands = []
    for width in widths:
        width = float(width)
        if not (np.isfinite(width) and 0 < width <= hi - lo):
            raise ValueError('Every band width must be in (0, k_range span]')
        starts = np.arange(lo, hi-width+1.e-10, step)
        starts = np.unique(np.r_[starts, hi-width])
        for left in starts:
            bands.append((float(left), float(left+width)))
    return np.asarray(bands)


def eigenvalues_from_covariance(k, tau, C, *, counts=None,
                                k_range=(0.5, 3.0),
                                widths=(0.35, 0.5, 0.75), step=0.1,
                                lag_index=1, min_modes=20,
                                max_condition=1.e6, min_modulus=0.25,
                                min_cycles=0.5, duration=None):
    """Estimate bandwise rates from C(k,tau) and scan over physical k bands.

    For each band, A=C(tau)@inv(C(0)), with eigenvalues mu, and
    lambda=log(mu)/tau.  The 2x2 real propagator is estimated from the
    *mode-count weighted* covariances (counts=None gives equal bin weights).
    In the stable linear regime the result estimates the response generator.
    In a nonlinear or non-stationary regime it is an effective local fit.

    IMPORTANT: C must be computed WITHOUT temporal demeaning for an evolving
    state.  Pre-existing C_k_tau files might have been time demeaned, in
    which case inferred rates may be biased.  Spatial demeaning is fine.
    Direct estimators should use estimate_linear_eigenvalues below.

    Omega_max=max |Im(lambda)| and r_max=max Re(lambda) are INDEPENDENT
    maxima over the scanned bands. Missing/ill-conditioned bands are skipped.
    Small propagator eigenvalues (poor one-lag predictability) are rejected.
    If duration is supplied, oscillations with <min_cycles cycles over that
    duration are unresolved and receive Omega=0. These are heuristics, not
    formal hypothesis tests. No complex pair -> 0; no valid fits -> NaN.
    """
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    C = np.asarray(C)
    if k.ndim != 1 or tau.ndim != 1 or C.shape != (len(tau), len(k), 2, 2):
        raise ValueError('Require k[nk], tau[nt], C[nt,nk,2,2]')
    if lag_index <= 0 or lag_index >= len(tau) or not np.isclose(tau[0], 0):
        raise ValueError('lag_index must select a positive lag, and tau[0]=0')
    delta_t = float(tau[lag_index])
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError('Selected lag must be positive')
    if counts is None:
        weights = np.ones(len(k), dtype=float)
        # At least two independent radial bins for a pre-averaged C.
        minimum = 2
    else:
        weights = np.asarray(counts, dtype=float)
        if weights.shape != k.shape or np.any(weights < 0):
            raise ValueError('counts must be nonnegative and match k')
        minimum = int(min_modes)
    if (minimum < 1 or max_condition <= 1 or
        not 0 <= min_modulus < 1 or min_cycles < 0 or
        (duration is not None and duration <= 0)):
        raise ValueError('Invalid covariance-fit or resolution thresholds')

    bands = _scanning_bands(k_range, widths, step)
    omega = np.full(len(bands), np.nan)
    growth = np.full(len(bands), np.nan)
    cond = np.full(len(bands), np.nan)
    n_modes = np.zeros(len(bands), dtype=int)
    for i, (lo, hi) in enumerate(bands):
        selected = (k >= lo-1.e-12) & (k <= hi+1.e-12) & (weights > 0)
        selected &= np.isfinite(C[0]).all(axis=(1, 2))
        selected &= np.isfinite(C[lag_index]).all(axis=(1, 2))
        n_modes[i] = int(np.sum(weights[selected]))
        if n_modes[i] < minimum:
            continue
        w = weights[selected]
        # FFT +/-k pairs or real DCT modes give a real-space response matrix.
        C0 = np.average(C[0, selected].real, axis=0, weights=w)
        C1 = np.average(C[lag_index, selected].real, axis=0, weights=w)
        C0 = 0.5 * (C0 + C0.T)
        cond[i] = np.linalg.cond(C0)
        if not np.isfinite(cond[i]) or cond[i] > max_condition:
            continue
        try:
            A = np.linalg.solve(C0.T, C1.T).T
        except np.linalg.LinAlgError:
            continue
        # Real 2x2: exclude nonphysical negative real discrete eigenvalues,
        # for which the principal complex logarithm would invent pi/dt.
        disc = (A[0, 0]-A[1, 1])**2 + 4.*A[0, 1]*A[1, 0]
        if disc < 0:
            mu = np.linalg.eigvals(A.astype(complex))
            if (not np.all(np.isfinite(mu)) or
                np.min(np.abs(mu)) < min_modulus or np.any(np.abs(mu) <= 0)):
                continue
            rates = np.log(mu)/delta_t
            omega[i] = float(np.max(np.abs(rates.imag)))
            if (duration is not None and
                omega[i]*duration < 2*np.pi*min_cycles):
                omega[i] = 0.
            growth[i] = float(np.max(rates.real))
        else:
            mu = np.linalg.eigvalsh(A) if np.allclose(A, A.T) else np.linalg.eigvals(A)
            if (np.any(~np.isfinite(mu)) or
                np.any(np.real(mu) <= 0) or np.max(np.real(mu)) < min_modulus):
                continue
            omega[i] = 0.
            growth[i] = float(np.max(np.log(np.real(mu)))/delta_t)

    good_omega = np.flatnonzero(np.isfinite(omega))
    good_growth = np.flatnonzero(np.isfinite(growth))
    good_complex = np.flatnonzero((omega > 0) & np.isfinite(growth))
    i_omega = int(good_omega[np.argmax(omega[good_omega])]) if len(good_omega) else None
    i_growth = int(good_growth[np.argmax(growth[good_growth])]) if len(good_growth) else None
    i_complex_growth = (int(good_complex[np.argmax(growth[good_complex])])
                        if len(good_complex) else None)
    def center(i):
        return float(bands[i].mean()) if i is not None else np.nan
    return dict(Omega_max=float(omega[i_omega]) if i_omega is not None else np.nan,
                r_max=float(growth[i_growth]) if i_growth is not None else np.nan,
                r_at_Omega_max=float(growth[i_omega]) if i_omega is not None else np.nan,
                r_complex_max=(float(growth[i_complex_growth])
                               if i_complex_growth is not None else np.nan),
                Omega_at_r_complex_max=(float(omega[i_complex_growth])
                                        if i_complex_growth is not None else np.nan),
                k_r_complex_max=center(i_complex_growth),
                k_Omega_max=center(i_omega), k_r_max=center(i_growth),
                band_Omega_max=bands[i_omega].copy() if i_omega is not None else np.full(2,np.nan),
                band_r_max=bands[i_growth].copy() if i_growth is not None else np.full(2,np.nan),
                bands=bands, omega_by_band=omega, r_by_band=growth,
                condition_by_band=cond, n_modes_by_band=n_modes,
                lag_time=delta_t)


def estimate_linear_eigenvalues(phi_runs, dt_saved, L=(50., 50.), *,
                                bc='Neumann', k_range=(0.2, 2.5),
                                widths=(0.25, 0.35, 0.5), step=0.05,
                                bin_width=0.1, lag_frames=1,
                                burn_in=0, end_frame=None,
                                min_modes=20, max_condition=1.e6,
                                min_modulus=0.25, min_cycles=0.5):
    """Fit dynamic eigenvalues from raw runs without temporal demeaning.

    Accepts (2,T,Nx,Ny) or (R,2,T,Nx,Ny). Multiple runs are pooled at
    the covariance level *before* fitting, reducing maximal-band selection
    bias. The domain L sets physical k=pi*n/L for Neumann DCT modes.
    dt_saved is the interval BETWEEN SAVED snapshots, not the solver dt.
    Use time windows that remain approximately homogeneous and linear.
    """
    a = np.asarray(phi_runs)
    if a.ndim == 4:
        a = a[None, ...]
    if a.ndim != 5 or a.shape[1] != 2:
        raise ValueError('phi_runs must have shape (2,T,Nx,Ny) or (R,2,T,Nx,Ny)')
    if end_frame is None:
        end_frame = a.shape[2]
    if not (0 <= burn_in < end_frame <= a.shape[2] and
            end_frame-burn_in > lag_frames >= 1):
        raise ValueError('Need at least lag_frames+1 frames after burn_in')
    if bin_width <= 0 or not np.isfinite(bin_width):
        raise ValueError('bin_width must be positive')
    hi = float(k_range[1])
    edges = np.arange(0., hi+bin_width, bin_width)
    if edges[-1] < hi:
        edges = np.r_[edges, hi]
    # Match the source times in C0 and C1.  In particular, an estimator
    # using an all-times C(0) and a shorter-lag C(dt) can severely bias
    # growing (nonstationary) modes. Both use t=0,...,T-lag-1 here.
    geo = _geometry(a.shape[-2:], L, bc, n_bins=len(edges)-1,
                    bin_edges=edges, k_max=None)
    nb = len(geo['counts'])
    cov = np.zeros((2, nb, 2, 2), dtype=complex)
    n_pairs = end_frame - burn_in - lag_frames
    for run in a:
        z = _modes(run[:, burn_in:end_frame], geo)
        X, Y = z[:, :-lag_frames], z[:, lag_frames:]
        for aa in range(2):
            for bb in range(2):
                for lag, Z in enumerate((X, Y)):
                    per_mode = np.einsum('tm,tm->m', Z[aa], X[bb].conj(),
                                         optimize=True) / n_pairs
                    cov[lag, :, aa, bb] += geo['area_cell'] * _bin_complex(
                        per_mode, geo['bins'], geo['counts'], nb)
    cov /= len(a)
    return eigenvalues_from_covariance(
        geo['centers'], [0., lag_frames*float(dt_saved)], cov,
        counts=geo['counts'], k_range=k_range, widths=widths,
        step=step, lag_index=1, min_modes=min_modes,
        max_condition=max_condition, min_modulus=min_modulus,
        min_cycles=min_cycles,
        duration=(end_frame-burn_in-1)*float(dt_saved))

def compute_observables(phi_run, L=(50.,50.), *, dt_saved=None, bc='periodic',
                        reference='white', n_bins=40, k_max=None,
                        k_definition='physical',
                        k_band=None,cell_capacity=None,burn_in=0,
                        lags=None,compute_dynamics=True,
                        compute_eigenmodes=False, eigen_kwargs=None,
                        demean_time=False, corr_method='fft',
                        mode_chunk_size=256, reference_cell_capacity=None,
                        reference_rho=None,  h=None):
    """Convenient all-in-one analysis; pure NumPy/Scipy, no solver state.

    `dt_saved` MUST be the separation of stored phi_run frames, i.e.
    dt_integrator*(nsteps/no_frames) when frames are equally spaced.
    With k_max=None the supported spectral range is inferred automatically
    from L, shape(phi_run) and bc. For nonlinear phase classification, you
    may optionally restrict k_max/k_band to large-scale patterns. Supplied
    lags too large for the frames remaining after burn_in are omitted with
    a warning; lags=None always works. To add the direct (matched-pair)
    eigenvalue estimate set compute_eigenmodes=True and optionally pass
    eigen_kwargs (for k range, widths, lag, etc.). Pass lags='all' to save
    ALL lags after burn_in, using the FFT algorithm (recommended).
    Explicit demean_time=False keeps the un-demeaned dynamics consistent
    with the direct propagator estimator; the spatial mean is always removed.
    For reference='beta0', reference_cell_capacity specifies how many
    microscopic exclusion sites are averaged per simulation grid cell.
    """
    a=_frames(phi_run)
    out=dict(excess=excess_spectral_power(a,L,reference=reference,bc=bc,
                     n_bins=n_bins,k_max=k_max,k_band=k_band,
                     reference_cell_capacity=reference_cell_capacity,
                     reference_rho=reference_rho, k_definition=k_definition, h=h),
             segregation=segregation_index(a,cell_capacity=cell_capacity,
                                           subtract_mixing=cell_capacity is not None),
             occupation=occupation_index(a))
    if compute_dynamics and a.shape[1]>2:
        if dt_saved is None: raise ValueError('dt_saved needed for dynamic observables')
        kw=dict(bc=bc,n_bins=n_bins,k_max=k_max,burn_in=burn_in,
                k_definition=k_definition, demean_time = demean_time)
        out['time_lag']=connected_time_lag_covariance(a,dt_saved,L,lags=lags,**kw)
        out['spatiotemporal']=spatiotemporal_structure_factor(a,dt_saved,L,**kw)
        out['time_lag']=connected_time_lag_covariance(
            a,dt_saved,L,lags=lags,
            method=corr_method,mode_chunk_size=mode_chunk_size,**kw)
        out['spatiotemporal']=spatiotemporal_structure_factor(
            a,dt_saved,L,**kw)
        if compute_eigenmodes:
            out['eigenmodes']=estimate_linear_eigenvalues(
                a,dt_saved,L=L,bc=bc,burn_in=burn_in,**(eigen_kwargs or {}))
    return out
