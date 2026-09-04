import numpy as np
import scipy as sp
import numba as nb


def fraction_unhappy(theta, phi):
    fractions = phi/phi.sum(axis=0)
    unhappy = phi[(theta-fractions)>0]
    if np.any(unhappy):
        return unhappy.sum()/phi.sum()
    else:
        return 0
    
def dissimilarity(phi):
    phi0 = 1 - np.sum(phi,axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    D = np.mean(np.abs(phi[0] - global_dist[0]))/global_dist[0]
    D += np.mean(np.abs(phi[1] - global_dist[1]))/global_dist[1]
    D += np.mean(np.abs(phi0 - global_dist[2]))/global_dist[2]
    return D/2

def mean_relative_entropy(phi):
    phi0 = 1 - np.sum(phi, axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    
    # Ensure no division by zero or log of zero
    phi_combined = np.vstack([phi[0], phi[1], phi0]).reshape((3,)+ phi0.shape)
    global_dist = np.clip(global_dist, 1e-10, None)
    phi_combined = np.clip(phi_combined, 1e-10, None)
    Sglobal = - np.sum(global_dist * np.log(global_dist))
    
    kl_divergence = np.sum(phi_combined * np.log(phi_combined / global_dist.reshape((3,)+ len(phi0.shape)*(1,))), axis=0)
    mean_kl_divergence = np.mean(kl_divergence)/Sglobal
    
    return mean_kl_divergence

def entropy_index(phi):
    '''
    Computes the entropy index of a field configuration. The entropy index is computed as:

    S_H = 1/H \sum_i f_i h_i

    where:
        i is the label for the discretized region
        f_i is the local fraction of occupants in the region
        h_i is the relative entropy between the local population composition and the global composition
        H is the Shannon entropy of the global composition
    '''
    phi_tot = np.sum(phi, axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean()])/np.sum(phi, axis = 0).mean()
    
    # Ensure no division by zero or log of zero
    global_dist = np.clip(global_dist, 1e-10, None)
    p_clipped = np.clip(phi/phi.sum(axis=0), 1e-10, None)
    H = - np.sum(global_dist * np.log(global_dist))
    
    kl_divergence = np.sum(p_clipped * np.log(p_clipped / global_dist.reshape((2,)+ len(phi_tot.shape)*(1,))), axis=0)
    
    f_i = phi_tot/np.sum(phi_tot)
    S_H = 1/H*np.sum(f_i*kl_divergence)
    return S_H

def measure_mode_growth(sim, rho_bar, param, kx_index, ky_index):
    Nx, Ny = sim.N
    x = np.arange(Nx)[:, None]
    y = np.arange(Ny)[None, :]

    mode = np.cos(2*np.pi*kx_index*x/Nx + 2*np.pi*ky_index*y/Ny)

    eps = 1e-6

    phi0 = np.empty((2, Nx, Ny))
    phi0[0] = rho_bar
    phi0[1] = rho_bar

    # composition/polarization perturbation
    phi = phi0.copy()
    phi[0] += eps * mode
    phi[1] -= eps * mode

    work = sim._ensure_work(phi.dtype)

    rhs = sim.rhs_SchellingwithVoter(
        phi, param, dt=1.0, toggle_noise=False, work=work
    ).copy()

    # project RHS onto the perturbation direction
    amp_rhs = np.sum((rhs[0] - rhs[1]) * mode) / np.sum(mode**2)

    # because perturbation in rho_A-rho_B has amplitude 2 eps
    growth_rate = amp_rhs / (2 * eps)

    return growth_rate


def makeD_second_order(Nx, dx, bc="periodic"):
    Dx = np.zeros((Nx, Nx))

    if bc == "periodic":
        for i in range(Nx):
            Dx[i, (i - 1) % Nx] = -0.5
            Dx[i, (i + 1) % Nx] = 0.5

    elif bc == "Neumann":
        for i in range(Nx):
            Dx[i, np.abs(i - 1)] += -0.5
            Dx[i, Nx - 1 - np.abs(Nx - 1 - (i + 1))] += 0.5

    else:
        raise ValueError(
            f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann'"
        )

    return sp.sparse.csc_array(Dx / dx)

def makeD2_second_order(Nx, dx, bc="periodic"):
    D2x = np.zeros((Nx, Nx))

    if bc == "periodic":
        for i in range(Nx):
            D2x[i, (i - 1) % Nx] = 1.0
            D2x[i, i] = -2.0
            D2x[i, (i + 1) % Nx] = 1.0

    elif bc == "Neumann":
        for i in range(Nx):
            D2x[i, np.abs(i - 1)] += 1.0
            D2x[i, i] += -2.0
            D2x[i, Nx - 1 - np.abs(Nx - 1 - (i + 1))] += 1.0

    else:
        raise ValueError(
            f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann'"
        )

    return sp.sparse.csc_array(D2x / dx**2)


def makeD3_second_order(Nx, dx, bc="periodic"):
    D3x = np.zeros((Nx, Nx))

    if bc == "periodic":
        for i in range(Nx):
            D3x[i, (i - 2) % Nx] = -0.5
            D3x[i, (i - 1) % Nx] = 1.0
            D3x[i, (i + 1) % Nx] = -1.0
            D3x[i, (i + 2) % Nx] = 0.5

    elif bc == "Neumann":
        for i in range(Nx):
            D3x[i, np.abs(i - 2)] += -0.5
            D3x[i, np.abs(i - 1)] += 1.0
            D3x[i, Nx - 1 - np.abs(Nx - 1 - (i + 1))] += -1.0
            D3x[i, Nx - 1 - np.abs(Nx - 1 - (i + 2))] += 0.5

    else:
        raise ValueError(
            f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann'"
        )

    return sp.sparse.csc_array(D3x / dx**3)

def makeD(Nx,dx, bc = "periodic"):
    Dx = np.zeros((Nx, Nx))
    # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	\
    if bc == 'periodic':
        for i in range(Nx):
            Dx[i, (i-4) % Nx] = 1/280
            Dx[i, (i-3) % Nx] = -4/105
            Dx[i, (i-2) % Nx] = 1/5
            Dx[i, (i-1) % Nx] = -4/5
            Dx[i, (i+1) % Nx] = 4/5
            Dx[i, (i+2) % Nx] = -1/5
            Dx[i, (i+3) % Nx] = 4/105
            Dx[i, (i+4) % Nx] = -1/280
    elif bc == "Neumann":
        # Use that u(-dx)=u(dx), u(-2dx)=u(2dx), such that Dx[i, -1] = Dx[i, 1] etc. so use absolute value
        # On the other boundary it implies Dx[Nx] = Dx[Nx-2] (zero indexing, boundary is at Nx-1), so map Nx + k to Nx - 1 - |Nx-1 - (i+k)|
        for i in range(Nx):    
            Dx[i, np.abs(i-4)] += 1/280
            Dx[i, np.abs(i-3)] += -4/105
            Dx[i, np.abs(i-2)] += 1/5
            Dx[i, np.abs(i-1)] += -4/5
            Dx[i, Nx -1 - np.abs(Nx -1 - (i+1))] += 4/5
            Dx[i, Nx -1 - np.abs(Nx -1 - (i+2))] += -1/5
            Dx[i, Nx -1 - np.abs(Nx -1 - (i+3))] += 4/105
            Dx[i, Nx -1 - np.abs(Nx -1 - (i+4))] += -1/280
    else:
        raise ValueError(f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann' ")
    return sp.sparse.csc_array(Dx/dx)

def makeD2(Nx, dx, bc = 'periodic'):
    D2x = np.zeros((Nx,Nx))
    # −1/560	8/315	−1/5	8/5	−205/72	8/5	−1/5	8/315	−1/560	
    if bc == 'periodic':
        for i in range(Nx):
            D2x[i, (i-4) % Nx] = -1/560
            D2x[i, (i-3) % Nx] = 8/315
            D2x[i, (i-2) % Nx] = -1/5
            D2x[i, (i-1) % Nx] = 8/5
            D2x[i,i] = -205/72
            D2x[i, (i+1) % Nx] = 8/5
            D2x[i, (i+2) % Nx] = -1/5
            D2x[i, (i+3) % Nx] = 8/315
            D2x[i, (i+4) % Nx] = -1/560
    elif bc == 'Neumann':
        for i in range(Nx):
            D2x[i, np.abs(i-4)] += -1/560
            D2x[i, np.abs(i-3)] += 8/315
            D2x[i, np.abs(i-2)] += -1/5
            D2x[i, np.abs(i-1)] += 8/5
            D2x[i,i] += -205/72
            D2x[i,  Nx -1 - np.abs(Nx -1 - (i+1))] += 8/5
            D2x[i,  Nx -1 - np.abs(Nx -1 - (i+2))] += -1/5
            D2x[i,  Nx -1 - np.abs(Nx -1 - (i+3))] += 8/315
            D2x[i,  Nx -1 - np.abs(Nx -1 - (i+4))] += -1/560
    else:
        raise ValueError(f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann' ")
    return sp.sparse.csc_array(D2x/dx**2)

def makeD2_fv_neumann(N, dx):
    rows = []
    cols = []
    data = []

    inv_dx2 = 1.0 / dx**2

    # left boundary cell
    rows += [0, 0]
    cols += [0, 1]
    data += [-inv_dx2, inv_dx2]

    # interior cells
    for i in range(1, N - 1):
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        data += [inv_dx2, -2.0 * inv_dx2, inv_dx2]

    # right boundary cell
    rows += [N - 1, N - 1]
    cols += [N - 2, N - 1]
    data += [inv_dx2, -inv_dx2]

    return sp.sparse.csc_matrix((data, (rows, cols)), shape=(N, N))

def makeD3(Nx, dx, bc = 'periodic'):
    D3x = np.zeros((Nx,Nx))
    # −7/240	3/10	−169/120	61/30	0	−61/30	169/120	−3/10	7/240	
    if bc == 'periodic':
        for i in range(Nx):
            D3x[i, (i-4) % Nx] = -7/240
            D3x[i, (i-3) % Nx] = 3/10
            D3x[i, (i-2) % Nx] = -169/120
            D3x[i, (i-1) % Nx] = 61/30
            # D3[i,i] = 0
            D3x[i, (i+1) % Nx] = -61/30
            D3x[i, (i+2) % Nx] = 169/120
            D3x[i, (i+3) % Nx] = -3/10
            D3x[i, (i+4) % Nx] = 7/240
    elif bc == 'Neumann':
        for i in range(Nx):
            D3x[i, np.abs(i-4)] += -7/240
            D3x[i, np.abs(i-3)] += 3/10
            D3x[i, np.abs(i-2)] += -169/120
            D3x[i, np.abs(i-1)] += 61/30
            # D3[i,i] = 0
            D3x[i, Nx -1 - np.abs(Nx -1 - (i+1))] += -61/30
            D3x[i, Nx -1 - np.abs(Nx -1 - (i+2))] += 169/120
            D3x[i, Nx -1 - np.abs(Nx -1 - (i+3))] += -3/10
            D3x[i, Nx -1 - np.abs(Nx -1 - (i+4))] += 7/240
    else:
        raise ValueError(f"Boundary conditions {bc} not implemented, try 'periodic' or 'Neumann' ")
    return sp.sparse.csc_array(D3x/dx**3)

def makeD4(Nx, dx):
    D4x = np.zeros((Nx,Nx))
    # 7/240	−2/5	169/60	−122/15	91/8	−122/15	169/60	−2/5	7/240	
    for i in range(Nx):
        D4x[i, (i-4) % Nx] = 7/240
        D4x[i, (i-3) % Nx] = -2/5
        D4x[i, (i-2) % Nx] = 169/60
        D4x[i, (i-1) % Nx] = -122/15
        D4x[i,i] = 91/8
        D4x[i, (i+1) % Nx] = -122/15
        D4x[i, (i+2) % Nx] = 169/60
        D4x[i, (i+3) % Nx] = -2/5
        D4x[i, (i+4) % Nx] = 7/240
    return sp.sparse.csc_array(D4x/dx**4)

def power_spectrum(phi_run, L,  num_bins=60, averaged = True, bp = 0, centered = True):
    '''Compute the angle averaged power spectrum for a run, averaged over timeseries from until the end bp:'''
    Nx, Ny = phi_run.shape[2:]
    Lx, Ly = L
    dx, dy = Lx / Nx, Ly / Ny

    if not averaged:
        bp = -1

    if centered:
        fields = phi_run[:,bp:,:,:] - phi_run[:,bp:,:,:].mean(axis=(2,3))[:,:,np.newaxis,np.newaxis]
        phi_empty = 1 - phi_run[:,bp:,:,:].sum(axis=0)
        phi_empty = phi_empty - phi_empty.mean(axis=(1,2))[:,np.newaxis,np.newaxis]
    else:
        fields = phi_run[:,bp:,:,:]
        phi_empty = 1 - phi_run[:,bp:,:,:].sum(axis=0)
        
    fields = np.concatenate((fields, phi_empty[np.newaxis,:,:,:]), axis=0)

    # Fourier Transform
    phi_k = np.fft.fft2(fields, axes=(2, 3))

    # Power Spectrum
    power_spectrum = np.einsum("atij, atij -> atij", np.conjugate(phi_k), phi_k).real
    G_AB = (np.conjugate(phi_k[0])*phi_k[1]).real
    # G_BA = (np.conjugate(phi_k[1])*phi_k[0]).real
    
    # average along time axis
    power_spectrum = power_spectrum.mean(axis=1)
    G_AB = G_AB.mean(axis=0)

    # Compute wave numbers
    kx = np.fft.fftfreq(Nx, d=dx)*2*np.pi
    ky = np.fft.fftfreq(Ny, d=dy)*2*np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)

    # Angle-Averaged Power Spectrum
    k_flat = k.ravel()
    k_bins = np.linspace(0, np.max(k), num_bins)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    power_spectra = np.zeros((power_spectrum.shape[0], len(k_bin_centers)))
    G = np.zeros(len(k_bin_centers))
    
    for a in range(power_spectrum.shape[0]):
        power_spectrum_flat = power_spectrum[a].ravel()
        G_AB_flat = G_AB.ravel()
        for i in range(len(k_bin_centers)):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            power_spectra[a,i] = np.mean(power_spectrum_flat[mask])
            G[i] = np.mean(G_AB_flat[mask])
        
    # return k_bin_centers, power_spectra
    return k_bin_centers, power_spectra, G
    
def power_spectrum_1d(phi_run, L,  num_bins=50, averaged = True, bp = 0):
    '''Compute the angle averaged power spectrum for a run, averaged over timeseries from until the end bp:
    1D version of the code above'''
    N = phi_run.shape[2]
    dx = L / N

    if not averaged:
        bp = -1
        
    fields = phi_run[:,bp:,:]
    phi_empty = 1 - phi_run[:,bp:,:].sum(axis=0)
    fields = np.concatenate((fields, phi_empty[np.newaxis,:,:]), axis=0)

    fields_fluct = fields - np.mean(fields, axis = (1,2))[:,np.newaxis,np.newaxis]
    # Fourier Transform
    phi_k = np.fft.fft(fields_fluct)

    # Power Spectrum
    power_spectrum = np.abs(phi_k)**2
    S_ab = np.mean((phi_k[0]*np.conj(phi_k[1])).real, axis=0)
    # average along time axis
    power_spectrum = power_spectrum.mean(axis=1)

    # Compute wave numbers
    kx = np.fft.fftfreq(N, d=dx)*2*np.pi
    k = np.sqrt(kx**2)

    # Angle-Averaged Power Spectrum
    k_flat = k.ravel()
    k_bins = np.linspace(0, np.max(k), num_bins)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    S_ab_spectrum = np.zeros(len(k_bin_centers))
    power_spectra = np.zeros((power_spectrum.shape[0], len(k_bin_centers)))

    for a in range(power_spectrum.shape[0]):
        power_spectrum_flat = power_spectrum[a].ravel()
        for i in range(len(k_bin_centers)):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            power_spectra[a,i] = np.mean(power_spectrum_flat[mask])
    
    S_ab_flat = S_ab.ravel()
    for i in range(len(k_bin_centers)):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        S_ab_spectrum[i] = np.mean(S_ab_flat[mask])
        

    return k_bin_centers, power_spectra, S_ab_spectrum

def fit_amplitude_logspace(theory, data):
    """
    Fit a multiplicative amplitude amp in log-space:
        log(data) ~ log(amp) + log(theory)
    """
    valid = (
        np.isfinite(theory) & np.isfinite(data) &
        (theory > 0) & (data > 0)
    )

    if not np.any(valid):
        return 1.0

    log_amp = np.mean(np.log(data[valid]) - np.log(theory[valid]))
    return np.exp(log_amp) 

from scipy.fft import dctn


def power_spectrum_dct_2d(
    phi_run,
    L,
    bp=0,
    centered=True,
    num_bins=50,
    averaged=True,
    use_lattice_k=True,
):
    """
    2D power spectrum using the DCT appropriate for cell-centered
    Neumann finite-volume fields.

    Parameters
    ----------
    phi_run : array, shape (2, nframes, Nx, Ny)
        Density history.

    L : tuple
        (Lx, Ly)

    bp : int
        First frame included in time averaging.

    centered : bool
        Subtract spatial mean independently from each frame.

    num_bins : int
        Number of radial bins.

    averaged : bool
        If False, use only the final frame.

    use_lattice_k : bool
        False:
            radial coordinate is continuum
                k = sqrt(kx^2 + ky^2)

        True:
            radial coordinate is the FV lattice wavenumber
                khat^2 = 4/dx^2 sin^2(kx dx/2)
                       + 4/dy^2 sin^2(ky dy/2)

    Returns
    -------
    k_centers : (nbins,) array

    spectra : (3, nbins) array
        AA, BB, vacancy auto-spectra.

    G_AB : (nbins,) array
        AB cross-spectrum.

    power_2d : (3, Nx, Ny) array
        Time-averaged 2D DCT power.

    G_AB_2d : (Nx, Ny) array
        Time-averaged 2D AB cross-spectrum.

    k_2d : (Nx, Ny) array
        Radial wavenumber used for binning.
    """

    Lx, Ly = L
    _, nframes, Nx, Ny = phi_run.shape

    dx = Lx / Nx
    dy = Ly / Ny

    if not averaged:
        fields = phi_run[:, -1:, :, :].copy()
    else:
        fields = phi_run[:, bp:, :, :].copy()

    # Vacancy field
    phi0 = 1.0 - fields.sum(axis=0)

    fields = np.concatenate(
        (fields, phi0[np.newaxis, ...]),
        axis=0,
    )

    # ------------------------------------------------------------
    # Remove the DCT zero mode independently at each time.
    # ------------------------------------------------------------
    if centered:
        fields -= fields.mean(
            axis=(2, 3),
            keepdims=True,
        )

    # ------------------------------------------------------------
    # DCT-II in spatial dimensions.
    #
    # norm="ortho" gives an orthonormal transform, convenient for
    # comparisons between resolutions.
    # ------------------------------------------------------------
    coeff = dctn(
        fields,
        type=2,
        axes=(2, 3),
        norm="ortho",
    )

    # Auto spectra
    power_2d_t = coeff**2

    # AB cross spectrum
    G_AB_t = coeff[0] * coeff[1]

    # Average over recorded frames
    power_2d = power_2d_t.mean(axis=1)
    G_AB_2d = G_AB_t.mean(axis=0)

    # ------------------------------------------------------------
    # Neumann/DCT mode numbers.
    #
    # Continuum eigenmodes:
    #     k_n = pi n / L
    # ------------------------------------------------------------
    nx = np.arange(Nx)
    ny = np.arange(Ny)

    kx = np.pi * nx / Lx
    ky = np.pi * ny / Ly

    # ------------------------------------------------------------
    # Either continuum radial k or exact FV lattice khat.
    # ------------------------------------------------------------
    if use_lattice_k:
        kx_used = (
            2.0 / dx
            * np.sin(0.5 * kx * dx)
        )
        ky_used = (
            2.0 / dy
            * np.sin(0.5 * ky * dy)
        )
    else:
        kx_used = kx
        ky_used = ky

    KX, KY = np.meshgrid(
        kx_used,
        ky_used,
        indexing="ij",
    )

    k_2d = np.sqrt(KX**2 + KY**2)

    # ------------------------------------------------------------
    # Radial binning
    # ------------------------------------------------------------
    k_flat = k_2d.ravel()

    bins = np.linspace(
        0.0,
        k_flat.max(),
        num_bins + 1,
    )

    k_centers = 0.5 * (bins[:-1] + bins[1:])

    spectra = np.full(
        (3, num_bins),
        np.nan,
    )

    G_AB = np.full(num_bins, np.nan)

    for i in range(num_bins):

        mask = (
            (k_flat >= bins[i])
            & (k_flat < bins[i + 1])
        )

        # Include upper endpoint in final bin
        if i == num_bins - 1:
            mask |= k_flat == bins[-1]

        if not np.any(mask):
            continue

        for a in range(3):
            spectra[a, i] = np.mean(
                power_2d[a].ravel()[mask]
            )

        G_AB[i] = np.mean(
            G_AB_2d.ravel()[mask]
        )

    return (
        k_centers,
        spectra,
        G_AB,
        power_2d,
        G_AB_2d,
        k_2d,
    )


def check_convergence(Obs_list, T, eps_mean = 1e-3, K = 50):
    """ Checks whether the change in running averages of the mean and Fano factors of all observables in Obs_list 
    are below the specified eps_mean and eps_Fano for the last K samples.
    """
    criteria = []
    for Obs in Obs_list:
        Obs_sliding = np.array([np.mean(Obs[t:t+T]) for t in range(len(Obs)-T)])
        # var_sliding = np.array([np.var(Obs[t:t+T]) for t in range(len(Obs)-T)])
        # Fano = var_sliding/Obs_sliding

        mean_check = np.abs(Obs_sliding[1:]-Obs_sliding[:-1])/Obs_sliding[:-1]
        criteria.append(mean_check[-K:] < eps_mean)

        # Fano_check = np.abs(Fano[1:]-Fano[:-1])/Fano[:-1] 
        # criteria.append(Fano[-K:] < eps_Fano)
            
    return np.all(criteria)

def kmean(power_spectrum, k_bins):
    '''Computes the mean of the wavenumber averaged over the power spectrum as distribution'''
    Z = np.sum(power_spectrum)
    kmean = np.sum(k_bins*power_spectrum)
    return kmean/Z


def voter_interface_density(phi_run, eps=1e-14):
    """
    Interface-density diagnostic used by Dornic, Chate & Munoz.

    Parameters
    ----------
    phi_run : ndarray, shape (2, nframes, Nx, Ny)
        phi_run[0] = rho_A
        phi_run[1] = rho_B

    Returns
    -------
    rho_I : ndarray, shape (nframes,)
        rho_I(t) = 1 - <m(r,t) m(r+e,t)>,
        averaged over x- and y-nearest-neighbor bonds.
    """

    rho_A = phi_run[0]
    rho_B = phi_run[1]

    n = rho_A + rho_B

    # Voter composition / magnetization field in [-1, 1].
    # For pure voter with n=1 this is simply rho_A - rho_B.
    m = np.divide(
        rho_A - rho_B,
        n,
        out=np.zeros_like(rho_A),
        where=n > eps,
    )

    # Nearest-neighbor correlations in the two lattice directions.
    Cx = np.mean(
        m * np.roll(m, -1, axis=1),
        axis=(1, 2),
    )

    Cy = np.mean(
        m * np.roll(m, -1, axis=2),
        axis=(1, 2),
    )

    C_nn = 0.5 * (Cx + Cy)

    return 1.0 - C_nn