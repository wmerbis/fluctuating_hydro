import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import trange
from numba import njit, prange, typed

# --- Parameters ---
repetition = 1   # animation will be heavy, use 1 repetition
Lx = 50
Ly = 50
N = Lx * Ly
gamma = 0.2
#beta = 10
vacant = 0.3
kappa_aa = 1
kappa_bb = 1
kappa_ab = 1
kappa_ba = -1
lambda_a =0.1
lambda_b =0.1
Gamm=0.1
# --- Graph / lattice generation ---
def generate_graph(Lx, Ly):
    N = Lx*Ly
    A = np.zeros((N, N), np.uint8)
    neighbors = np.zeros((N, 4), dtype=int)

    def node(x, y): return x + Lx * y

    for x in range(Lx):
        for y in range(Ly):
            i = node(x, y)
            right = node((x+1)%Lx, y)
            left  = node((x-1)%Lx, y)
            up    = node(x, (y+1)%Ly)
            down  = node(x, (y-1)%Ly)
            neighbors[i] = [right, left, up, down]
            for j in neighbors[i]:
                A[i,j] = A[j,i] = 1
    return A, neighbors

# # --- Utility difference ---
@njit(parallel=True)
def utility_and_hypothetical_utility_flat(x, neighbors,
                                          kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamm):
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
        s_plus = 0
        s_minus = 0
        for k in range(z):
            j = neighbors[i, k]
            s_plus += n_plus[j]
            s_minus += n_minus[j]
        laplacian_plus[i] = -4.0 * n_plus[i] + s_plus
        laplacian_minus[i] = -4.0 * n_minus[i] + s_minus

    # 3) compute pi_plus, pi_minus, pi
    pi_plus = np.empty(N, dtype=np.float64)
    pi_minus = np.empty(N, dtype=np.float64)
    pi = np.empty(N, dtype=np.float64)
    for i in prange(N):
        pi_plus[i] = kappa_aa * n_plus[i] + kappa_ab * n_minus[i] + Gamm*laplacian_plus[i]
        pi_minus[i] = kappa_ba * n_plus[i] + kappa_bb * n_minus[i] + Gamm*laplacian_minus[i]
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
                if x[i] == 1:
                    delta_flat[pos] = pi_plus[j] - pi[i]
                else:
                    # x[i] == -1 expected (we skipped x[i]==0 above)
                    delta_flat[pos] = pi_minus[j] - pi[i]
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
def compute_move_rates_flat(delta_flat, gamma):
    """
      - rates: 1d array of acceptance rates (len == len(delta_flat))
    """
    total_moves = delta_flat.shape[0]
    rates = np.empty(total_moves, dtype=np.float64)
    beta = 2.0 / gamma
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

# --- Simulation ---
@njit
def simulation_with_snapshots(time, y0, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b , snapshot_interval=5000):
    t = time[0]
    tmax = time[-1]
    x = y0.copy()
    snapshots = [x.copy().reshape(Ly,Lx)]  # store initial state
    step = 0
    while t < tmax:
        delta_flat, vac_indices, vac_starts, owner, voter_reaction = utility_and_hypothetical_utility_flat(x.astype(np.int64),
                                                                                    neighbors.astype(np.int64),
                                                                                    kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamm)

        rates = compute_move_rates_flat(delta_flat,  gamma)
        voter_rates = compute_voter_rates(voter_reaction, lambda_a, lambda_b)
        # Combine into one array
        total_len = rates.shape[0] + voter_rates.shape[0]
        rates_new = np.empty(total_len, dtype=np.float64)
        for i in range(rates.shape[0]):
            rates_new[i] = rates[i]
        for i in range(voter_rates.shape[0]):
            rates_new[rates.shape[0] + i] = voter_rates[i]

        total_rates = np.sum(rates_new)
        if total_rates == 0:
            break
        #selected_move = np.random.choice(len(rates), p=rates/total_rates) #O(N)
        #method with #O(log(N))
        # r = np.random.rand()             
        # threshold = r * total_rates
        # selected_move = np.searchsorted(np.cumsum(rates), threshold)
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
            snapshots.append(x.copy().reshape(Ly,Lx))
    return snapshots

#generate initially a pattern to look for the travelling phase
from scipy.ndimage import gaussian_filter1d

def make_wave_pattern(Lx, Ly, M,freq_x, freq_y, smoothness):
    # if freq_x and freq_y large enough, we get patches 
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi*freq_x, Lx), np.linspace(0, 2*np.pi*freq_y, Ly))
    wave = np.sin(X + 0.5*Y) + 0.5*np.sin(1.5*X - 0.3*Y) + 0.2*np.random.randn(Ly, Lx)
    smooth = gaussian_filter1d(wave, sigma=smoothness, axis=1)
    flat =smooth.flatten()
    y0 = np.zeros_like(flat)
    abs_sorted_indices = np.argsort(np.abs(flat))
    # Indices of elements that will be nonzero
    nonzero_indices = abs_sorted_indices[M:]  
    # Assign ±1 based on sign of original values
    y0[nonzero_indices] = np.sign(flat[nonzero_indices])
    return y0.flatten()

# --- Run a single simulation for animation ---
A, neighbors = generate_graph(Lx, Ly)
#random initial condition
y0 = np.zeros(N)
nonzero_indices = np.random.choice(N, N-int(vacant*N), replace=False)
y0[nonzero_indices] = np.random.choice([-1,1], size=N-int(vacant*N))

# warm up
x0 = y0.copy()  # or generate a small random x
_ = utility_and_hypothetical_utility_flat(x0.astype(np.int64), neighbors.astype(np.int64),
                                          kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamm)

if (kappa_ab*kappa_ba) < 0:
    #prepatterned condition
    y0 = make_wave_pattern(Lx, Ly,int(vacant*N), freq_x=4, freq_y=4, smoothness=3)
snapshots = simulation_with_snapshots(np.linspace(0,  100*N, 1000*N), y0, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b)

#snapshots = simulation_with_snapshots_jit(20000*N, y0, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba, 1000, Lx, Ly)

# --- Animation ---
fig, ax = plt.subplots()
cmap = plt.get_cmap('bwr', 3)  # blue-white-red
im = ax.imshow(snapshots[0], cmap=cmap, vmin=-1, vmax=1)
ax.set_title(fr"$\Gamma = {Gamm},\, D = {gamma/2},\, \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, \lambda_a={lambda_a}, \, \lambda_b={lambda_b} $")
plt.colorbar(im, ax=ax, ticks=[-1,0,1], label='State')

def update(frame):
    im.set_array(snapshots[frame])
    return [im]

anim = FuncAnimation(fig, update, frames=len(snapshots), interval=5000, blit=True)
# --- Export as GIF ---
gif_writer = PillowWriter(fps=10)
anim.save("with_voter_reaction.gif", writer=gif_writer)
plt.show()
