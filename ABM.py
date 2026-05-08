import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import trange
from numba import njit, prange, typed
from numpy.fft import fft2, ifft2, fftshift
import time
FS_TITLE = 22
FS_LABEL = 20
FS_TICK  = 20
FS_CBAR  = 20
# ---------------- Parameters ----------------------
where_to_start =5
repetition = 1   #  use 1 repetition for animation
M = 1800   # number of snapshots to average over
Nx = 128
Ny = 128
N = Nx * Ny
max_step=2000*N
n_discard = 100   # discard first 100 saved snapshots
snapshot_interval = N # snapshot_interval = max(1, max_step // M)# max_step/M 
Da = 1
DA = Da   # diffusion of species A
DB = Da   # diffusion of species B
beta = 10
betaA = beta
betaB = beta
vacant = 0.3
kappa_aa = 0.6
kappa_bb = 0.6
kappa_ab = -0.4
kappa_ba = -0.4
Dv = 0.01 
lambda_a = Dv
lambda_b = Dv
Gamm=1
Gamma_aa = Gamm
Gamma_bb = Gamm
Gamma_ab = 0 
Gamma_ba = 0

# -------------------------define physical k axes ----------------------------------
# (if domain lengths Lx,Ly, and spacing dx and dy)
Lx=12.8
h = Lx/Nx # for the theory comparison
dx = Lx/Nx # in general, if we know the physical length Lx, and the number of houses along one axis, Nx then dx= Lx/Nx, the same for dy=Ly/Ny 
dy = Lx/Nx
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Nx, d=dy)
kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
kgrid = np.sqrt(kxg**2 + kyg**2)
k_flat = kgrid.ravel()

# -------------------- radial binning procedure ---------------------------------

kmax = k_flat.max()
nbins = min(60, Nx//2)   # choose reasonable number bins
kbins = np.linspace(0.0, kmax, nbins+1)
kcent = 0.5*(kbins[:-1] + kbins[1:])
bin_index = np.searchsorted(kbins, k_flat) - 1
valid_bin = (bin_index >= 0) & (bin_index < nbins)
counts = np.bincount(bin_index[valid_bin], minlength=nbins)

mask = counts > 0

# k grid excluding zero mode and any bins with zero count
k_vals = kcent[mask]

# we pick indices where k>0 (should be all valid except possibly first)
pos      = k_vals > 0
if len(pos) == 0:
    raise RuntimeError("No nonzero k bins.")

k_use = k_vals[pos]
k_vals   = k_use

#----------------------------------------helper functions--------------------------

def radial_average_2d(S2d):
    Sflat = S2d.ravel().real
    Srad = np.bincount(
        bin_index[valid_bin],
        weights=Sflat[valid_bin],
        minlength=nbins
    ).astype(float)
    Srad[mask] /= counts[mask]
    return Srad

def fit_power_law_time(t, y):
    """
    Fit y ~ C * t^slope on log-log axes.
    Returns:
        alpha  such that y ~ C * t^(-alpha)
        C
        slope  in log(y) = slope*log(t) + intercept
    """
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    t_fit = t[mask]
    y_fit = y[mask]

    if len(t_fit) < 2:
        return np.nan, np.nan, np.nan

    logt = np.log(t_fit)
    logy = np.log(y_fit)

    slope, intercept = np.polyfit(logt, logy, 1)
    C = np.exp(intercept)
    alpha = -slope
    return alpha, C, slope

# --- Log–log slope extraction ---
def loglog_slope(k, S):
    mask = (k > 0) & (S > 0)
    k = k[mask]
    S = S[mask]
    if len(k) < 2:
        return np.nan, np.nan
    logk = np.log(k)
    logS = np.log(S)
    A = np.vstack([logk, np.ones(len(k))]).T
    sol, *_ = np.linalg.lstsq(A, logS, rcond=None)
    slope, intercept = sol
    return -slope, intercept

#generate initially a pattern to look for the travelling phase

def make_wave_pattern(Nx, Ny, M,freq_x, freq_y, smoothness):
    # if freq_x and freq_y large enough, we get patches 
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi*freq_x, Nx), np.linspace(0, 2*np.pi*freq_y, Ny))
    wave = np.sin(X + 0.5*Y) + 0.5*np.sin(1.5*X - 0.3*Y) + 0.2*np.random.randn(Ny, Nx)
    smooth = gaussian_filter1d(wave, sigma=smoothness, axis=1)
    flat =smooth.flatten()
    y0 = np.zeros_like(flat)
    abs_sorted_indices = np.argsort(np.abs(flat))
    # Indices of elements that will be nonzero
    nonzero_indices = abs_sorted_indices[M:]  
    # Assign ±1 based on sign of original values
    y0[nonzero_indices] = np.sign(flat[nonzero_indices])
    return y0.flatten()

#----------------------------------Jay's prediction-------------------------------
# === Drift matrix M(k) and noise matrix Q(k) ===
def noise_matrix(k, a0, b0, h):
    """Return the 2x2 noise covariance Q(k) = Q0 + Q2 k^2 in 2D."""
    k2 = k**2
    r0 = 1-a0-b0
    q11 = DA * h**2 * r0 * a0 * k2 + 2.0 * Dv * a0* b0
    q22 = DB * h**2 * r0 * b0 * k2 + 2.0 * Dv * a0 * b0
    q12 = -2.0 * Dv * a0* b0
    return np.array([[q11, q12],
                     [q12, q22]], dtype=float)

def drift_matrix(k, a0, b0):
    """Return the 2x2 drift matrix M(k) = A2 k^2 + A4 k^4."""
    k2 = k**2
    k4 = k2**2
    r0 = 1-a0-b0
    
    Maa = (DA*(r0 + a0) + Dv*b0 - DA*betaA*r0*a0*kappa_aa) * k2    + DA*betaA*r0*a0*Gamma_aa * k4
    
    Mab = (DA*a0 - Dv*a0 - DA*betaA*r0*a0*kappa_ab) * k2           + DA*betaA*r0*a0*Gamma_ab * k4
    
    Mba = (DB*b0 - Dv*b0 - DB*betaB*r0*b0*kappa_ba) * k2           + DB*betaB*r0*b0*Gamma_ba * k4
    
    Mbb = (DB*(r0 + b0) + Dv*a0 - DB*betaB*r0*b0*kappa_bb) * k2    + DB*betaB*r0*b0*Gamma_bb * k4
    
    return np.array([[Maa, Mab],
                     [Mba, Mbb]], dtype=float)

def solve_lyapunov_2x2(M, R):
    """Solve M S + S M^T = R for symmetric 2x2 S.
    
    M = [[a,b],[c,d]]
    S = [[x,y],[y,z]]
    R = [[R11,R12],[R12,R22]]
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    R11, R12, R22 = R[0, 0], R[0, 1], R[1, 1]
    
    # Linear system:
    # a x + b y = R11/2
    # c y + d z = R22/2
    # c x + (a + d) y + b z = R12
    A = np.array([
        [a,     b,   0.0],
        [0.0,   c,   d  ],
        [c, a + d,   b  ]
    ], dtype=float)
    rhs = np.array([R11/2.0, R22/2.0, R12], dtype=float)
    
    x, y, z = np.linalg.solve(A, rhs)
    return np.array([[x, y],
                     [y, z]], dtype=float)

def structure_factor(k, a0, b0, h):
    """Full equal-time structure factor S(k) solving M(k) S + S M^T = 2 Q(k)."""
    M = drift_matrix(k, a0, b0)
    Q = noise_matrix(k, a0, b0, h)
    R = 2.0 * Q
    return solve_lyapunov_2x2(M, R)

def SAA_th(k, a0, b0, h):
    return structure_factor(k, a0, b0, h)[0, 0]

def SAB_th(k, a0, b0, h):
    return structure_factor(k, a0, b0, h)[0, 1]

def SBB_th(k, a0, b0, h):
    return structure_factor(k, a0, b0, h)[1, 1]

#----------------------------------------Agent-based simulation--------------------------
# --- Graph / lattice generation ---
def generate_graph(Nx, Ny):
    N = Nx*Ny
    A = np.zeros((N, N), np.uint8)
    neighbors = np.zeros((N, 4), dtype=int)

    def node(x, y): return x + Nx * y

    for x in range(Nx):
        for y in range(Ny):
            i = node(x, y)
            right = node((x+1)%Nx, y)
            left  = node((x-1)%Nx, y)
            up    = node(x, (y+1)%Ny)
            down  = node(x, (y-1)%Ny)
            neighbors[i] = [right, left, up, down]
            for j in neighbors[i]:
                A[i,j] = A[j,i] = 1
    return A, neighbors

@njit(parallel=True)
def voter_only(x, neighbors):
    """
    this function returns voter dynamics only:
    """
    N = x.shape[0]
    z = neighbors.shape[1]

    # 1) Count neighbor types & empty counts
    n_plus = np.zeros(N, dtype=np.int64)
    n_minus = np.zeros(N, dtype=np.int64)
    empty_counts = np.zeros(N, dtype=np.int64)
    total_pairs = 0
    for i in prange(N):
        plus = 0
        minus = 0
        empty = 0
        for k in range(z):
            j = neighbors[i, k]
            v = x[j]
            if v == 1:
                plus += 1
            elif v == -1:
                minus += 1
                if x[i] == 1:
                    total_pairs += 1
            else:
                # v == 0
                empty += 1
        n_plus[i] = plus
        n_minus[i] = minus
        empty_counts[i] = empty

    # 1.1) make the 1-d array for all ordered pair of x[i] = 1 (origin) and x[j] =-1 (target)
    voter_reaction = np.empty(2 * total_pairs, dtype=np.int64)
    p = 0
    for i in range(N):
        if x[i] == 1:
            for k in range(z):
                j = neighbors[i, k]
                if x[j] == -1:
                    voter_reaction[p] = i
                    voter_reaction[p + 1] = j
                    p += 2

    return  voter_reaction


# # --- Utility difference ---
@njit(parallel=True)
def utility_and_hypothetical_utility_flat(x, neighbors,
                                          kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb):
    """
    this function returns flattened move data:
      - delta_flat: float64[:]  (Δπ for each possible move)
      - owner: 1d int array where owner[pos] = origin node that owns flattened move pos
      - vac_indices: int64[:]   (destination node index for each move)
      - vac_starts: int64[:]    (prefix array of length N+1; that marks where each i's moves live in, namely, range(vac_starts[i], vac_starts[i+1]) 
    """
    N = x.shape[0]
    z = neighbors.shape[1]

    # 1) Count neighbor types & empty counts
    n_plus = np.zeros(N, dtype=np.int64)
    n_minus = np.zeros(N, dtype=np.int64)
    empty_counts = np.zeros(N, dtype=np.int64)
    total_pairs = 0
    for i in prange(N):
        plus = 0
        minus = 0
        empty = 0
        for k in range(z):
            j = neighbors[i, k]
            v = x[j]
            if v == 1:
                plus += 1
            elif v == -1:
                minus += 1
                if x[i] == 1:
                    total_pairs += 1
            else:
                # v == 0
                empty += 1
        n_plus[i] = plus
        n_minus[i] = minus
        empty_counts[i] = empty

    # 1.1) make the 1-d array for all ordered pair of x[i] = 1 (origin) and x[j] =-1 (target)
    voter_reaction = np.empty(2 * total_pairs, dtype=np.int64)
    p = 0
    for i in range(N):
        if x[i] == 1:
            for k in range(z):
                j = neighbors[i, k]
                if x[j] == -1:
                    voter_reaction[p] = i
                    voter_reaction[p + 1] = j
                    p += 2

    # 2) Compute Laplacians ---
    laplacian_plus = np.zeros(N, dtype=np.float64)
    laplacian_minus = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        if x[i] ==1:
            laplacian_plus[i] = -4.0 + n_plus[i]
            laplacian_minus[i] = n_minus[i]
        elif x[i] ==-1:
            laplacian_minus[i] = -4.0 + n_minus[i]
            laplacian_plus[i] = n_plus[i]
        else:
            laplacian_minus[i] =-4+ n_minus[i]
            laplacian_plus[i] =-4+ n_plus[i]

    # 3) compute pi_plus, pi_minus, pi
    pi_plus = np.empty(N, dtype=np.float64)
    pi_minus = np.empty(N, dtype=np.float64)
    pi = np.empty(N, dtype=np.float64)
    for i in prange(N):
        pi_plus[i] = kappa_aa * n_plus[i] + kappa_ab * n_minus[i] + Gamma_aa*laplacian_plus[i] +Gamma_ab*laplacian_minus[i]
        pi_minus[i] = kappa_ba * n_plus[i] + kappa_bb * n_minus[i] + Gamma_bb*laplacian_minus[i] +  Gamma_ba*laplacian_plus[i]
    for i in prange(N):
        xi = x[i]
        if xi == 1:
            pi[i] = pi_plus[i]
        elif xi == -1:
            pi[i] = pi_minus[i]
        else:
            pi[i] = 0.0

    # 3) Flatten all vacancies into arrays
    total_vac = 0 #total number of vacants of all the nodes
    for i in range(N):
        total_vac += empty_counts[i]

    vac_indices = np.empty(total_vac, dtype=np.int64)  # flattened index for all possible destination vacants
    delta_flat = np.empty(total_vac, dtype=np.float64) # Δπ per move
    owner      = np.empty(total_vac, dtype=np.int64)   # owner (origin) of the moves
    vac_starts = np.empty(N + 1, dtype=np.int64)

    pos = 0 #total number of vacants of all the "occupied" nodes
    for i in range(N):
        vac_starts[i] = pos
        if x[i] == 0:
            # if origin empty, we will not add moves for it
            continue
        # for each neighbor j that is empty, record a move i -> j
        for k in range(z):
            j = neighbors[i, k]
            if x[j] == 0:
                vac_indices[pos] = j
                owner[pos]    = i
                # contribution if moved to j depends on what spin i has
                if x[i] == 1: #we have not consider the new local configuration as the move happens
                    pi_plus_vacant = pi_plus[j] #- (kappa_aa + Gamma_aa)  # 5 comes from -4 *(n==1) and -1 of reducing 1 plus meighbor
                    delta_flat[pos] = pi_plus_vacant - pi[i]
                else:
                    # x[i] == -1 expected (we skipped x[i]==0 above)
                    pi_minus_vacant = pi_minus[j] # - (kappa_bb + Gamma_bb)
                    delta_flat[pos] = pi_minus_vacant- pi[i]
                pos += 1

    vac_starts[N] = pos
    # NOTE: as some origin nodes x[i] were empty, pos should often be fewer than total_vac
    # If pos < total_vac, we return only the filled prefix
    if pos < total_vac:
        # resize arrays to actual length pos (Numba supports creating new arrays)
        out_vac_indices = np.empty(pos, dtype=np.int64)
        out_delta_flat = np.empty(pos, dtype=np.float64)
        out_owner = np.empty(pos, dtype=np.int64)

        for i in range(pos):
            out_vac_indices[i] = vac_indices[i]
            out_delta_flat[i] = delta_flat[i]
            out_owner[i]      = owner[i]

        owner       =  out_owner
        vac_indices = out_vac_indices
        delta_flat = out_delta_flat
    return delta_flat, vac_indices, vac_starts,  owner, voter_reaction

@njit(parallel=True)
def compute_move_rates_flat(delta_flat, beta, Da):
    """
      - rates: 1d array of acceptance rates (len == len(delta_flat))
    """
    total_moves = delta_flat.shape[0]
    rates = np.empty(total_moves, dtype=np.float64)
    gamma = 2.0 * Da
    rates= gamma / (1.0 + np.exp(-(beta) * delta_flat))
    return rates

@njit(parallel=True)
def compute_voter_rates(voter_reaction, lambda_a, lambda_b):
    """
    - Compute acceptance rates for all neighbor voter reactions.
    - voter_reaction : int64[:] (flattened [i0, j0, i1, j1, ...])
    - Returns
    -------
    voter_rates : float64[:]
        Flattened array with 2 entries per (i,j) pair:
        [rate_for_i_flip, rate_for_j_flip, ...]
    """
    n_pairs = voter_reaction.shape[0] // 2
    voter_rates = np.empty(2 * n_pairs, dtype=np.float64)

    for p in prange(n_pairs):
        # origin and target indices
        i = voter_reaction[2 * p]      # so voter_reaction[0::2] → all origin nodes (x=1)
        j = voter_reaction[2 * p + 1]  # so voter_reaction[1::2] → all corresponding targets (x=-1)

        # assign rates (for every ordered pair (i,j), we store 1 entries in voter_rates)
        # p is the index of the pair in voter_reaction
        voter_rates[2 * p]     = lambda_b  # i: x[i] = 1 → -1 so origin node flips with probability λ_b
        voter_rates[2 * p + 1] = lambda_a  # j: x[j] = -1 → 1 so target node flips with probability λ_a

    return voter_rates

# --- Main Simulation ---
@njit
def simulation_with_snapshots(time, y0, neighbors, beta, Da, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b , snapshot_interval=snapshot_interval, Schelling=True):
    t = time[0]
    x = y0.copy()
    snapshots = [x.copy().reshape(Ny,Nx)]  # store initial state
    snap_times = [t]
    step = 0
    while step < max_step:
        
        if Schelling:
            delta_flat, vac_indices, vac_starts, owner, voter_reaction = utility_and_hypothetical_utility_flat(x.astype(np.int64),
                                                                                        neighbors.astype(np.int64),
                                                                                        kappa_aa, kappa_bb, kappa_ab, kappa_ba,  Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb)

            rates = compute_move_rates_flat(delta_flat, beta, Da)
            voter_rates = compute_voter_rates(voter_reaction, lambda_a, lambda_b)
            # Combine into one array
            total_len = rates.shape[0] + voter_rates.shape[0]
            rates_new = np.empty(total_len, dtype=np.float64)
            for i in range(rates.shape[0]):
                rates_new[i] = rates[i]
            for i in range(voter_rates.shape[0]):
                rates_new[rates.shape[0] + i] = voter_rates[i]
        else:
            voter_reaction = voter_only(x.astype(np.int64),neighbors.astype(np.int64))
            voter_rates = compute_voter_rates(voter_reaction, lambda_a, lambda_b)
            total_len =  voter_rates.shape[0]
            rates_new = np.empty(total_len, dtype=np.float64)
            for i in range(voter_rates.shape[0]):
                rates_new[i] = voter_rates[i]


        total_rates = np.sum(rates_new)
        if total_rates == 0:
            break
        r_thresh = np.random.rand() * total_rates
        cumulative = 0.0
        selected_move = 0
        for i in range(rates_new.shape[0]):
            cumulative += rates_new[i]
            if cumulative >= r_thresh:
                selected_move = i
                break
        #Update
        if selected_move < rates.shape[0]:
            node_i = owner[selected_move]
            move_to = vac_indices[selected_move]   # destination node directly from flattened array
            x[move_to] = x[node_i]
            x[node_i] = 0
        else:
            v_index = selected_move - len(rates) # v_index is the index within the voter_rates array
            pair_index = v_index // 2
            i = voter_reaction[2 * pair_index]
            j = voter_reaction[2 * pair_index + 1]

            if v_index % 2 == 0:
                # Origin-side flip: +1 → -1
                x[i] = -1
            else:
                # Target-side flip: -1 → +1
                x[j] = 1

        t += -np.log(np.random.rand()) / total_rates
        step += 1
        if step % snapshot_interval == 0:
            snapshots.append(x.copy().reshape(Ny,Nx))
            snap_times.append(t)
    return snapshots,  np.array(snap_times), t

# ------------------------------------ Run a single simulation  --------------------

A, neighbors = generate_graph(Nx, Ny)
#random initial condition
y0 = np.zeros(N)
nonzero_indices = np.random.choice(N, N-int(vacant*N), replace=False)
y0[nonzero_indices] = np.random.choice([-1,1], size=N-int(vacant*N))

# warm up
x0 = y0.copy()  # or generate a small random x
_ = utility_and_hypothetical_utility_flat(x0.astype(np.int64), neighbors.astype(np.int64),
                                          kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb)

if (kappa_ab*kappa_ba) < 0:
    #prepatterned condition
    y0 = make_wave_pattern(Nx, Ny,int(vacant*N), freq_x=4, freq_y=4, smoothness=3)

t0 = time.time()
snapshots, snap_times, trun = simulation_with_snapshots(np.linspace(0,  N/10, N), y0, neighbors,  beta, Da, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b)
t1 = time.time()
print(trun)
print(f"Getting patterns in {t1-t0:.2f}s")

#--------------------------------------------------- Animation -------------------------------

fig, ax = plt.subplots()
cmap = plt.get_cmap('bwr', 3)  # blue-white-red
im = ax.imshow(snapshots[0], cmap=cmap, vmin=-1, vmax=1)
ax.set_title(fr"$\Gamma = {Gamm},\,  \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, Da = {Da},\, Dv={Dv}$")
plt.colorbar(im, ax=ax, ticks=[-1,0,1], label='State')

def update(frame):
    im.set_array(snapshots[frame])
    return [im]

anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000, blit=True)
# --- Export as GIF ---
gif_writer = PillowWriter(fps=10)
anim.save(f"Gamma{Gamm}_kappa_aa{kappa_aa}_kappa_ab{kappa_ab}_kappa_ba{kappa_ba}_Dv{Dv}_DA{Da}_movie.gif", writer=gif_writer)
plt.show()

# Pick the final snapshot and build densities 
final = snapshots[-1]          # shape (Ny, Nx)

#------------------------------Power spectrum computed over M snapshots-------------------------------

snapshots_analysis = snapshots[n_discard:]
snap_times_analysis = snap_times[n_discard:]

take = min(M, len(snapshots_analysis))  # take the mininum between the required last M snapshots and the real length of snapshot

S_AA_acc = np.zeros((Ny, Nx), dtype=np.float64)
S_BB_acc = np.zeros((Ny, Nx), dtype=np.float64)
S_AB_acc = np.zeros((Ny, Nx), dtype=np.float64)
S_00_acc = np.zeros((Ny, Nx), dtype=np.float64)

S_AA_tidx = np.full(( take, nbins), np.nan)
S_BB_tidx = np.full(( take, nbins), np.nan)
S_AB_tidx = np.full(( take, nbins), np.nan)
S_00_tidx = np.full(( take, nbins), np.nan)

# store characteristic wave number vs time
kmean_tidx = np.full((take), np.nan)
kpeak_tidx = np.full((take), np.nan)


for jj,snap in enumerate(snapshots_analysis[-take:]):
    #indicator fields
    rhoA = (snap == 1).astype(float)
    rhoB = (snap == -1).astype(float)
    rho0 = (snap == 0).astype(float)

    stack_field = np.stack([rhoA, rhoB])
    #H_field += mean_relative_entropy(stack_field)
    #D_field += dissimilarity(stack_field)
    
    #mean (global) densities
    a0 = rhoA.mean()
    b0 = rhoB.mean()

    #fluctuations
    dA = rhoA - a0
    dB = rhoB - b0
    d0 = rho0 - rho0.mean()

    # FFTs
    dA_k = fft2(dA)
    dB_k = fft2(dB)
    d0_k = fft2(d0)

    S_AA_2d = (dA_k * np.conj(dA_k)).real / N
    S_BB_2d = (dB_k * np.conj(dB_k)).real / N
    S_00_2d = (d0_k * np.conj(d0_k)).real / N
    S_AB_2d = np.real(dA_k * np.conj(dB_k)) / N # complex in general; take real part for radial averaging

    # accumulate structure factors over the last M snapshots
    S_AA_acc += S_AA_2d
    S_BB_acc += S_BB_2d 
    S_00_acc += S_00_2d
    S_AB_acc += S_AB_2d 

    # radial average at THIS time point
    S_AA_t  = radial_average_2d(S_AA_2d)
    S_BB_t  = radial_average_2d(S_BB_2d)
    S_00_t  = radial_average_2d(S_00_2d)
    S_AB_t  = radial_average_2d(S_AB_2d)

    S_AA_tidx[jj, :] = S_AA_t 
    S_BB_tidx[jj, :] = S_BB_t 
    S_00_tidx[jj, :] = S_00_t 
    S_AB_tidx[jj, :] = S_AB_t 

    # characteristic k at this time
    Ssym_t = S_AA_t[mask][pos] + S_BB_t[mask][pos]
    if np.any(np.isfinite(Ssym_t)) and np.nansum(Ssym_t) > 0:
        kmean_tidx[jj] = np.sum(k_use * Ssym_t) / np.sum(Ssym_t)
        kpeak_tidx[jj] = k_use[np.argmax(Ssym_t)]

# average over snapshots
S_AA_k = S_AA_acc / take
S_BB_k = S_BB_acc / take
S_AB_k = S_AB_acc / take
S_00_k = S_00_acc / take

# --------------------------------------Slope of domain growth -------------------------------

t_plot = snap_times_analysis[-take:]  # snapshot index; use real saved times if available
valid_t = ~np.isnan(kmean_tidx)

t_data = t_plot[valid_t]
k_data = kmean_tidx[valid_t]

plt.figure()
fit_mask = (t_data >= 10) & (t_data <= 5000)
t_fit = t_data[fit_mask]
k_fit_time = k_data[fit_mask]
alpha_fit, C_fit, slope_fit = fit_power_law_time(t_fit, k_fit_time)

if np.isfinite(alpha_fit):
    t_line = np.linspace(t_fit.min(), t_fit.max(), 200)
    k_line = C_fit * t_line**(slope_fit)
    plt.loglog(
        t_line,
        k_line,
        '--',
        linewidth=2,
        label=fr'fit: $\langle k\rangle \sim t^{{-{alpha_fit:.3f}}}$'
    )

if len(t_data) > 0:
    c1 = k_data[0] * t_data[0]**(1/3)
    c2 = k_data[0] * t_data[0]**(1/4)
    plt.loglog(t_data, c1 * t_data**(-1/3), label=r'$\sim t^{-1/3}$')
    plt.loglog(t_data, c2 * t_data**(-1/4), label=r'$\sim t^{-1/4}$')

plt.loglog(t_plot[valid_t], kmean_tidx[valid_t],'-o', label="simulations")

plt.xlabel("t")
plt.ylabel(r'$\langle k \rangle$')
plt.legend()
plt.title(fr"$ Da = {Da},\, Dv = {Dv} $")
plt.tight_layout()
#plt.show()
plt.savefig(f"Da{Da}_Dv{Dv}.png", dpi=300)  

#------------------------------------Power spectrum analysis---------------------------------
# flatten for radial averaging
S_AA_flat = S_AA_k.ravel().real
S_BB_flat = S_BB_k.ravel().real
S_00_flat = S_00_k.ravel().real
S_AB_flat = np.real(S_AB_k.ravel())
Srad_A = np.zeros(nbins)
Srad_B = np.zeros(nbins)
Srad_0 = np.zeros(nbins)
Srad_AB = np.zeros(nbins)

for i, k in enumerate(k_flat):
    binidx = np.searchsorted(kbins, k) - 1
    if 0 <= binidx < nbins:
        Srad_A[binidx] += S_AA_flat[i]
        Srad_B[binidx] += S_BB_flat[i]
        Srad_0[binidx] += S_00_flat[i]
        Srad_AB[binidx] += S_AB_flat[i]
        counts[binidx] += 1

mask = counts>0 
Srad_A[mask] /= counts[mask]
Srad_B[mask] /= counts[mask]
Srad_0[mask] /= counts[mask]
Srad_AB[mask] /= counts[mask]

# ------------------ find peak radial wavenumber ----------------
SA       = Srad_A[mask][pos]
SB       = Srad_B[mask][pos]
S0       = Srad_0[mask][pos]
SAB      =  Srad_AB[mask][pos]
S_sym    = SA + SB

k0 = k_vals[np.argmax(S_sym)]
print("Detected radial peak k0 =", k0,  "dx=",dx, " kmax=", kmax) 

#-------------------computing slope and fit--------------------------------------
# Use ONLY idx_low-th smallest and idx_high-th largest nonzero k bins:
# k[ pos[idx_low] : pos[-idx_high] ]  → includes everything in between but excludes the extreme ends

idx_low = np.argmax(S_sym)+where_to_start      # fifth smallest non-zero
idx_high = -2    # second largest non-zero
k_fit    = k_vals[idx_low:idx_high]
SA_fit   = SA[idx_low:idx_high]
SB_fit   = SB[idx_low:idx_high]
SAB_fit  = SAB[idx_low:idx_high]
S0_fit  = S0[idx_low:idx_high]

slope_A, _ = loglog_slope(k_fit, SA_fit)
slope_B, _ = loglog_slope(k_fit, SB_fit)
slope_0, _ = loglog_slope(k_fit, S0_fit)
slope_AB, _ = loglog_slope(k_fit, np.abs(SAB_fit))

print(f"log–log slope AA = {slope_A:.3f}")
print(f"log–log slope BB = {slope_B:.3f}")
print(f"log–log slope AB = {slope_AB:.3f}")

#------------------------------- saving --------------------------------------
#outfile = os.path.join(outdir, f"scan_Da_{Da:.3f}_Dv_{Dv:.3f}.npz")
np.savez(
    f"scan_Da_{Da:.3f}_Dv_{Dv:.3f}.npz",
    snapshots=snapshots,
    n_take=take,
    k0=k0,
    slopeA=slope_A,
    slopeB=slope_B,
    slopeAB=slope_AB,
    M= M,   # number of snapshots to keep for taking average, 
    Nx = Nx,
    max_step=max_step, 
    beta = beta,
    vacant = vacant, 
    kappa_ab=kappa_ab,
    kappa_ba=kappa_ba,
    kappa_aa = kappa_aa,
    kappa_bb = kappa_bb,
    Gamma=Gamm,
    h=h,  
)

# ------------------------------ plotting ------------------------------------
plt.figure(figsize=(14,6))

# 1) final snapshot
plt.subplot(1,3,1)
plt.title(fr"$\Gamma = {Gamm},\, Da = {Da},\, \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, Dv={Dv} $")
plt.imshow(final, cmap='bwr', vmin=-1, vmax=1, origin='lower')

# # 2) plt.ylabel('equal-time correlations')
plt.subplot(1,3,2)
plt.plot(k_vals, SA, 'o-', label='G_AA ', markersize=5)
plt.plot(k_vals, SB, 'v-', label='G_BB', markersize=5)
plt.plot(k_vals, SAB, 's-', label='G_AB', markersize=5)
plt.xlabel(r'$k$', fontsize=FS_LABEL)
plt.legend(fontsize=18)
plt.tick_params(axis='both', labelsize=FS_TICK)
plt.title(fr"$\rho_A^{(0)} = {a0:.2f}, \rho_B^{(0)} = {b0:.2f}$")
#plt.grid(True)

kplot = np.linspace(np.min(k_fit), np.max(k_fit), 200)
SAA_theory = np.array([SAA_th(k, a0, b0, h) for k in kplot])
SBB_theory = np.array([SBB_th(k, a0, b0, h) for k in kplot])
SAB_theory = np.array([SAB_th(k, a0, b0, h) for k in kplot])
# Optional: log-log view to check slope 
plt.subplot(1,3,3)
plt.loglog(k_fit, SA_fit, 'o', label=f'G_AA with 1/k^{slope_A:.3f}')
plt.loglog(kplot, SAA_theory, '--', label=f'theory for AA')
plt.loglog(k_fit, SB_fit, 'v', label=f'G_BB with 1/k^{slope_B:.3f}')
plt.loglog(kplot, SBB_theory, '--', label=f'theory for BB')
plt.loglog(k_fit,  np.abs(SAB_fit), 's', label=f'G_AB with 1/k^{slope_AB:.3f}')
if  kappa_ab == kappa_ba < 0 :
    plt.loglog(kplot, -SAB_theory, '-', label=f'theory for AB')
else:
    plt.loglog(kplot, SAB_theory, '-', label=f'theory for AB')
plt.xlabel('k')
plt.ylabel('G(k)')
plt.legend()
plt.title(
    fr"peak at: $ k0 = {k0:.3f}, h = {h}, dx = {dx}, Nx = {Nx}$")
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig(f"Gamma{Gamm}_kappa_aa{kappa_aa}_kappa_ab{kappa_ab}_kappa_ba{kappa_ba}_Dv{Dv}_DA{Da}.png", dpi=300)
plt.show()

# # 3) raw equal-time auto-correlation C_AA(x) (inverse FFT of full S_AA_k)
# C_AA_real = np.real(ifft2(S_AA_k))
# plt.subplot(1,3,2)
# plt.title("Equal-time C_AA(x) (centered)")
# plt.imshow(fftshift(C_AA_real), origin='lower')
# plt.colorbar()