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

def power_spectrum(phi_run, L,  num_bins=50, averaged = True, bp = 0):
    '''Compute the angle averaged power spectrum for a run, averaged over timeseries from until the end bp:'''
    Nx, Ny = phi_run.shape[2:]
    Lx, Ly = L
    dx, dy = Lx / Nx, Ly / Ny

    if not averaged:
        bp = -2
        
    fields = phi_run[:,bp:,:,:]
    phi_empty = 1 - phi_run[:,bp:,:,:].sum(axis=0)
    fields = np.concatenate((fields, phi_empty[np.newaxis,:,:,:]), axis=0)

    # Fourier Transform
    phi_k = np.fft.fft2(fields, axes=(2, 3))

    # Power Spectrum
    power_spectrum = np.abs(phi_k)**2
    # average along time axis
    power_spectrum = power_spectrum.mean(axis=1)

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
    
    for a in range(power_spectrum.shape[0]):
        power_spectrum_flat = power_spectrum[a].ravel()
        for i in range(len(k_bin_centers)):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            power_spectra[a,i] = np.mean(power_spectrum_flat[mask])
        
    return k_bin_centers, power_spectra
    