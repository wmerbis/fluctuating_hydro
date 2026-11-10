"""Optional low-k Neumann DCT recorder and temporal analysis for Schelling FHD.

Place as fhd/low_k_recording.py. The integrator only needs one call to
recorder.record(step, phi) at step 0 and at selected subsequent steps.
This module stores REAL orthonormal DCT-II modes; unlike periodic FFT modes,
Neumann DCT modes have no single traveling-wave propagation direction.
"""

import numpy as np
from scipy.fft import dctn, fft, fftfreq, ifft, next_fast_len


class LowKRecorder:
    """Retain selected physical-k amplitudes frequently and full fields rarely.

    All mode samples are uniformly spaced by mode_every * dt. The final
    integration state is NOT appended to mode_history unless it lands on
    that cadence, so its FFT timestamps remain uniformly spaced.
    """

    def __init__(self, N, L, nsteps, dt, *,
                 k_band=(0.2, 3.5), mode_every=100,
                 snapshot_every=10_000, dtype=np.float32):
        self.N = tuple(map(int, N))
        self.L = tuple(map(float, L))
        self.nsteps = int(nsteps)
        self.dt = float(dt)
        self.mode_every = int(mode_every)
        self.snapshot_every = (None if snapshot_every is None
                               else int(snapshot_every))
        self.k_band = tuple(map(float, k_band))
        if (len(self.N) != 2 or any(n < 2 for n in self.N)
                or any(x <= 0 for x in self.L) or self.nsteps < 1
                or self.dt <= 0 or self.mode_every < 1
                or (self.snapshot_every is not None and self.snapshot_every < 1)
                or not 0 <= self.k_band[0] < self.k_band[1]):
            raise ValueError("Invalid geometry, timestep, cadence, or band")
        if np.dtype(dtype) not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Recorder supports float32 or float64')
        self.dtype = np.dtype(dtype)
        kx = np.pi * np.arange(self.N[0]) / self.L[0]
        ky = np.pi * np.arange(self.N[1]) / self.L[1]
        k_grid = np.hypot(kx[:, None], ky[None, :])
        chosen = ((k_grid >= self.k_band[0]) &
                  (k_grid <= self.k_band[1]) & (k_grid > 0))
        self.mode_flat_indices = np.flatnonzero(chosen.ravel())
        self.k_modes = k_grid.ravel()[self.mode_flat_indices]
        if not len(self.k_modes):
            raise ValueError('No Neumann DCT modes in requested k band')
        n_mode = 1 + self.nsteps // self.mode_every
        self.mode_history = np.empty((2, n_mode, len(self.k_modes)),
                                     dtype=self.dtype)
        self.mode_steps = np.empty(n_mode, dtype=np.int64)
        self._imode = 0
        if self.snapshot_every is None:
            self.snapshots = None
            self.snapshot_steps = None
        else:
            n_fields = (1 + self.nsteps // self.snapshot_every
                        + int(self.nsteps % self.snapshot_every != 0))
            self.snapshots = np.empty((2, n_fields, *self.N), dtype=self.dtype)
            self.snapshot_steps = np.empty(n_fields, dtype=np.int64)
        self._ifield = 0

    @property
    def dt_saved(self):
        return self.mode_every * self.dt

    def needs_recording(self, step):
        return (step == 0 or step % self.mode_every == 0
                or (self.snapshot_every is not None and
                    (step % self.snapshot_every == 0 or step == self.nsteps)))

    def record(self, step, phi):
        """Call after the integration state is updated, with the true step."""
        step = int(step)
        if not 0 <= step <= self.nsteps:
            raise ValueError('Step outside trajectory')
        want_mode = (step == 0 or step % self.mode_every == 0)
        want_field = (self.snapshots is not None and
                      (step == 0 or step % self.snapshot_every == 0
                       or step == self.nsteps))
        if not (want_mode or want_field):
            return
        phi = np.asarray(phi)
        if phi.shape != (2, *self.N):
            raise ValueError(f'Expected density field shape {(2, *self.N)}')
        if want_mode:
            centered = phi - phi.mean(axis=(-2, -1), keepdims=True)
            transformed = dctn(centered, type=2, norm='ortho', axes=(-2, -1))
            self.mode_history[:, self._imode] = (
                transformed.reshape(2, -1)[:, self.mode_flat_indices])
            self.mode_steps[self._imode] = step
            self._imode += 1
        if want_field:
            self.snapshots[:, self._ifield] = phi
            self.snapshot_steps[self._ifield] = step
            self._ifield += 1

    def result(self):
        result = dict(
            mode_history=self.mode_history[:, :self._imode],
            mode_steps=self.mode_steps[:self._imode],
            k_modes=self.k_modes,
            mode_flat_indices=self.mode_flat_indices,
            dt_saved=self.dt_saved,
            L=np.asarray(self.L), N=np.asarray(self.N),
            k_band=np.asarray(self.k_band),
            area_cell=np.prod(self.L) / np.prod(self.N),
        )
        if self.snapshots is not None:
            result['snapshots'] = self.snapshots[:, :self._ifield]
            result['snapshot_steps'] = self.snapshot_steps[:self._ifield]
        return result


def _mode_bins(k_modes, k_edges):
    k_modes = np.asarray(k_modes, dtype=float)
    edges = np.asarray(k_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError('k_edges must be strictly increasing')
    ib = np.searchsorted(edges, k_modes, side='right') - 1
    valid = (ib >= 0) & (ib < len(edges) - 1)
    ib = ib[valid]
    counts = np.bincount(ib, minlength=len(edges)-1)
    return valid, ib, counts


def dynamic_from_modes(recording, *, burn_in=0, k_edges=None,
                       n_bins=40, demean_time=False, mode_chunk_size=128,
                       window='hann'):
    """Calculate full C(k,tau) and two-sided S(k,omega) from recorded DCT modes.

    Uses zero-padded temporal FFT for C, and unbiased lag normalization
    1/(T-lag). Long lags are returned but should be interpreted cautiously.
    """
    z0 = np.asarray(recording['mode_history'])
    k_modes = np.asarray(recording['k_modes'])
    dt_saved = float(recording['dt_saved'])
    area_cell = float(recording['area_cell'])
    if z0.ndim != 3 or z0.shape[0] != 2 or z0.shape[-1] != len(k_modes):
        raise ValueError('Expected mode_history shape (2,T,M)')
    if mode_chunk_size < 1:
        raise ValueError('mode_chunk_size must be >= 1')
    if not 0 <= burn_in < z0.shape[1] - 1:
        raise ValueError('burn_in must leave >=2 recorded mode frames')
    if k_edges is None:
        low = max(0., float(recording['k_band'][0]))
        high = float(recording['k_band'][1])
        k_edges = np.linspace(low, high * (1+1e-12), n_bins+1)
    edges = np.asarray(k_edges, dtype=float)
    valid, ib, counts = _mode_bins(k_modes, edges)
    z = z0[:, burn_in:, valid].astype(np.float64, copy=False)
    if demean_time:
        z = z - z.mean(axis=1, keepdims=True)
    T = z.shape[1]
    n_b = len(counts)
    lags = np.arange(T, dtype=int)
    nfft = next_fast_len(2*T-1)
    cross = np.zeros((n_b, 2, 2, nfft), dtype=np.complex128)
    for left in range(0, z.shape[-1], mode_chunk_size):
        right = min(z.shape[-1], left+mode_chunk_size)
        F = fft(z[:, :, left:right], n=nfft, axis=1)
        chosen_bins = ib[left:right]
        for b in np.unique(chosen_bins):
            sel = chosen_bins == b
            for a in range(2):
                for c in range(2):
                    cross[b,a,c] += np.sum(
                        F[a,:,sel]*np.conj(F[c,:,sel]),axis=0)
    C = np.full((T,n_b,2,2), np.nan, dtype=np.float64)
    for b in range(n_b):
        if counts[b]:
            raw = ifft(cross[b], axis=-1)[..., :T].real
            C[:,b] = np.moveaxis(raw / (T-lags)[None,None,:], -1, 0) * area_cell/counts[b]

    if window == 'hann':
        w = np.hanning(T)
    elif window in ('boxcar', None):
        w = np.ones(T)
    else:
        raise ValueError('window must be hann or boxcar')
    if not np.sum(w*w):
        raise ValueError('Too few frames for Hann window')
    scale = area_cell*dt_saved / np.sum(w*w)
    Smat = np.full((n_b,T,2,2), np.nan+0j, dtype=np.complex128)
    for left in range(0, z.shape[-1], mode_chunk_size):
        right = min(z.shape[-1], left+mode_chunk_size)
        F = fft(z[:,:,left:right]*w[None,:,None],axis=1)
        chosen_bins = ib[left:right]
        for b in np.unique(chosen_bins):
            sel = chosen_bins == b
            for a in range(2):
                for c in range(2):
                    block = (F[a,:,sel] * np.conj(F[c,:,sel])).sum(axis=0)
                    if not np.isfinite(Smat[b,0,a,c]):
                        Smat[b,:,a,c] = 0
                    Smat[b,:,a,c] += (scale/counts[b]) * block
    omega = 2*np.pi*fftfreq(T, d=dt_saved)
    sort = np.argsort(omega)
    centers = 0.5*(edges[:-1] + edges[1:])
    return dict(
        time_lag=dict(C=C, tau=lags*dt_saved, lags=lags, k=centers,
                      k_edges=edges, counts=counts, dt_saved=dt_saved,
                      n_frames_used=T, demean_time=demean_time),
        spatiotemporal=dict(S=Smat[:,sort], omega=omega[sort],
                            k=centers, counts=counts, k_edges=edges,
                            dt_saved=dt_saved, demean_time=demean_time),
    )


def eigenvalues_from_modes(recording, *, burn_in=0, end_frame=None,
                           lag_frames=1, bin_width=0.1, **kwargs):
    """Matched-time covariance estimator; delegates band scan to observables."""
    from fhd.observables import eigenvalues_from_covariance

    data = np.asarray(recording['mode_history'], dtype=np.float64)
    kmodes = np.asarray(recording['k_modes'])
    area_cell = float(recording['area_cell'])
    dt_saved = float(recording['dt_saved'])
    if end_frame is None:
        end_frame = data.shape[1]
    if not (0 <= burn_in < end_frame <= data.shape[1]
            and end_frame - burn_in > lag_frames >= 1):
        raise ValueError('Need enough selected mode frames')
    k_range = kwargs.get('k_range', (0.5,2.5))
    hi = float(k_range[1])
    edges = np.arange(0., hi + bin_width, bin_width)
    if edges[-1] < hi: edges = np.r_[edges,hi]
    valid, bins, counts = _mode_bins(kmodes, edges)
    z = data[:, burn_in:end_frame, valid]
    X,Y = z[:, :-lag_frames], z[:, lag_frames:]
    nb = len(counts)
    cov = np.full((2,nb,2,2),np.nan, dtype=np.float64)
    for a in range(2):
        for b in range(2):
            p0 = np.mean(X[a]*X[b],axis=0)
            p1 = np.mean(Y[a]*X[b],axis=0)
            for l,p in enumerate((p0,p1)):
                sums = np.bincount(bins, weights=p,minlength=nb)
                cov[l,counts>0,a,b] = area_cell*sums[counts>0]/counts[counts>0]
    return eigenvalues_from_covariance(
        (edges[:-1]+edges[1:])/2,
        [0.,lag_frames*dt_saved], cov, counts=counts,
        duration=(end_frame-burn_in-1)*dt_saved, **kwargs)
