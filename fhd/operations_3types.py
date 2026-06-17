import numpy as np
import scipy as sp


def fraction_unhappy(theta, phi):
    fractions = phi/phi.sum(axis=0)
    unhappy = phi[(theta-fractions)>0]
    if np.any(unhappy):
        return unhappy.sum()/phi.sum()
    else: 
        return 0
    
def dissimilarity(phi):
    ns = phi.shape[0]
    phi0 = 1 - np.sum(phi,axis=0)
    # global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    global_dist = np.array([phi[a].mean() for a in range(ns)] + [phi0.mean()])

    D = np.mean(np.abs(phi0 - global_dist[3]))/global_dist[3]
    # D += np.mean(np.abs(phi[1] - global_dist[1]))/global_dist[1]
    # D += np.mean(np.abs(phi0 - global_dist[2]))/global_dist[2]
    for a in range(ns):
        D += np.mean(np.abs(phi[a] - global_dist[a]))/global_dist[a]
    return D/ns

def mean_relative_entropy(phi):
    ns = phi.shape[0]
    phi0 = 1 - np.sum(phi, axis=0)
    # global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    global_dist = np.array([phi[a].mean() for a in range(ns)] + [phi0.mean()])
    
    # Ensure no division by zero or log of zero
    # phi_combined = np.vstack([phi[0], phi[1], phi0]).reshape((3,)+ phi0.shape)
    # phi_combined = np.vstack([phi, phi0]).reshape((ns,)+ phi0.shape)
    phi_combined = np.vstack([phi, phi0[None, :]])
    global_dist = np.clip(global_dist, 1e-10, None)
    phi_combined = np.clip(phi_combined, 1e-10, None)
    Sglobal = - np.sum(global_dist * np.log(global_dist))
    
    # kl_divergence = np.sum(phi_combined * np.log(phi_combined / global_dist.reshape((3,)+ len(phi0.shape)*(1,))), axis=0)
    kl_divergence = np.sum(phi_combined * np.log(phi_combined / global_dist.reshape((ns+1,)+ len(phi0.shape)*(1,))), axis=0) 
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
    ns = phi.shape[0]
    phi_tot = np.sum(phi, axis=0)
    # global_dist = np.array([phi[0].mean(), phi[1].mean(),phi[2].mean()])/np.sum(phi, axis = 0).mean()
    global_dist = np.array([phi[a].mean() for a in range(ns)])/np.sum(phi, axis = 0).mean()
    
    # Ensure no division by zero or log of zero
    global_dist = np.clip(global_dist, 1e-10, None)
    p_clipped = np.clip(phi/phi.sum(axis=0), 1e-10, None)
    H = - np.sum(global_dist * np.log(global_dist))
    
    kl_divergence = np.sum(p_clipped * np.log(p_clipped / global_dist.reshape((ns,)+ len(phi_tot.shape)*(1,))), axis=0)
    
    f_i = phi_tot/np.sum(phi_tot)
    S_H = 1/H*np.sum(f_i*kl_divergence)
    return S_H

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

def power_spectrum(phi_run, L,  num_bins=50, averaged = True, bp = 0, centered = True):
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


def power_spectrum(phi_run, L,  num_bins=50, averaged = True, bp = 0, centered = False):
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
    G_AC = (np.conjugate(phi_k[0])*phi_k[2]).real
    G_BC = (np.conjugate(phi_k[1])*phi_k[2]).real
    
    # G_BA = (np.conjugate(phi_k[1])*phi_k[0]).real
    
    # average along time axis
    power_spectrum = power_spectrum.mean(axis=1)
    G_AB = G_AB.mean(axis=0)
    G_AC = G_AC.mean(axis=0)
    G_BC = G_BC.mean(axis=0)

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
    
    G_AB_k = np.zeros(len(k_bin_centers))
    G_AC_k = np.zeros(len(k_bin_centers))
    G_BC_k = np.zeros(len(k_bin_centers))

    G_AB_flat = G_AB.ravel()
    G_AC_flat = G_AC.ravel()
    G_BC_flat = G_BC.ravel()
    
    for a in range(power_spectrum.shape[0]):
        power_spectrum_flat = power_spectrum[a].ravel()

        for i in range(len(k_bin_centers)):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            power_spectra[a,i] = np.mean(power_spectrum_flat[mask])
            G_AB_k[i] = np.mean(G_AB_flat[mask])
            G_AC_k[i] = np.mean(G_AC_flat[mask])
            G_BC_k[i] = np.mean(G_BC_flat[mask])
        
    # return k_bin_centers, power_spectra
    return k_bins[:-1], power_spectra, G_AB_k, G_AC_k, G_BC_k

def power_spectrum_data(data, L, num_bins=50, centered=False):
    """
    Compute angle averaged power spectrum for data with shape (ns, N, N)
    so it can be compared directly to power_spectrum() from simulations.
    """

    ns, Nx, Ny = data.shape
    Lx, Ly = L
    dx, dy = Lx / Nx, Ly / Ny

    if centered:
        fields = data - data.mean(axis=(1,2))[:,None,None]
        phi_empty = 1 - data.sum(axis=0)
        phi_empty = phi_empty - phi_empty.mean()
    else:
        fields = data
        phi_empty = 1 - data.sum(axis=0)

    # add empty phase like in simulation function
    fields = np.concatenate((fields, phi_empty[None,:,:]), axis=0)

    # Fourier transform
    phi_k = np.fft.fft2(fields, axes=(1,2))

    # Power spectrum
    power_spectrum = (np.conjugate(phi_k) * phi_k).real

    # Cross spectra
    G_AB = (np.conjugate(phi_k[0]) * phi_k[1]).real
    G_AC = (np.conjugate(phi_k[0]) * phi_k[2]).real
    G_BC = (np.conjugate(phi_k[1]) * phi_k[2]).real

    # Wave numbers
    kx = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2*np.pi
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)

    # Radial binning
    k_flat = k.ravel()
    k_bins = np.linspace(0, np.max(k), num_bins)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    power_spectra = np.zeros((power_spectrum.shape[0], len(k_bin_centers)))
    G_AB_k = np.zeros(len(k_bin_centers))
    G_AC_k = np.zeros(len(k_bin_centers))
    G_BC_k = np.zeros(len(k_bin_centers))

    G_AB_flat = G_AB.ravel()
    G_AC_flat = G_AC.ravel()
    G_BC_flat = G_BC.ravel()

    for a in range(power_spectrum.shape[0]):
        power_flat = power_spectrum[a].ravel()

        for i in range(len(k_bin_centers)):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            power_spectra[a,i] = np.mean(power_flat[mask])

            if a == 0:  # compute cross spectra only once
                G_AB_k[i] = np.mean(G_AB_flat[mask])
                G_AC_k[i] = np.mean(G_AC_flat[mask])
                G_BC_k[i] = np.mean(G_BC_flat[mask])

    return k_bins[:-1], power_spectra, G_AB_k, G_AC_k, G_BC_k