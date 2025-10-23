import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import trange

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
kappa_ab = -1
kappa_ba = -1
b=0
d=0

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

# --- Utility difference ---
def utility_and_hypothetical_utility(x,neighbors, kappa_aa, kappa_bb, kappa_ab, kappa_ba):
    x_i = x[:, None]
    x_j = x[neighbors] # shape (N, z), z=number of neighbors

    # vacant neighbors
    m_zero = (x_j == 0)
    empty_neighbors = [neighbors[i][m_zero[i]] for i in range(len(x))]

    # non-vacant neighbors
    local_field_left = np.sum(x_j == 1, axis=1) 
    local_field_right = np.sum(x_j == -1, axis=1)

    m_i_left  = (x_i == 1)
    m_i_right = (x_i == -1)
    contrib = m_i_left * (kappa_aa*local_field_left[:, None]  + kappa_ab*local_field_right[:, None]) + m_i_right * (kappa_ba*local_field_left[:, None] + kappa_bb*local_field_right[:, None]) 
    #current utility
    pi = contrib.sum(axis=1)

    #hypothetical utility: contrib_hyp[i]: a list of arrays, each containing the computed hypothetical contributions for node i’s vacant neighbors.
    #one need to know the local fields of each of the vacant neighbor of i
    #So,for each node i, and for each vacant neighbor j ∈ empty_neighbors[i],
    #the hypothetical contribution (utility that i would get if it moved into that vacancy)
    #is evaluated with the local field at that vacant site using the same κ-coefficients 
    contrib_hyp = []
    delta_pi = [] 
    for i in range(N):
        vacs = empty_neighbors[i]
        if len(vacs) == 0 or x[i] == 0:
            contrib_hyp.append(np.empty(0))
            delta_pi.append(np.empty(0))
            continue
        
        # for each vacant site, count +1 and -1 among its own neighbors
        x_vn = x[neighbors[vacs]]   # shape (len(vacs), z)
        n_plus  = np.sum(x_vn == 1, axis=1)
        n_minus = np.sum(x_vn == -1, axis=1)
        contrib_h = (x[i] == 1) * (kappa_aa*n_plus + kappa_ab*n_minus) + (x[i] == -1)  * (kappa_ba*n_plus + kappa_bb*n_minus) 

        contrib_hyp.append(contrib_h)
        # hypothetical gain in utility
        delta = contrib_h - pi[i]
        delta_pi.append(delta)

    return delta_pi, empty_neighbors

def compute_move_rates(delta_pi, gamma):
    """
    Given delta_pi as list-of-arrays for each i,
    compute acceptance probabilities p_accept[i][k] for each move option,
    then flatten into a 1-d array `rates`.
    Also return an array `idx_map` of the same length that maps each flattened entry
    to a tuple (i, k).
    """
    entries = []
    idx_map = []  # list of (i, k) pairs
    for i, d_arr in enumerate(delta_pi):
        for k, d in enumerate(d_arr):
            beta = 2.0/gamma
            p_accept = gamma / (1 + np.exp(-(beta) * d))
            entries.append(p_accept)
            idx_map.append((i, k))

    rates = np.array(entries, dtype=float)
    return rates, idx_map

# --- Simulation ---
def simulation_with_snapshots(time, y0, A, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba, snapshot_interval=100):
    t, *_, tmax = time
    x = y0.copy()
    snapshots = [x.copy().reshape(Ly,Lx)]  # store initial state
    step = 0
    while t < tmax:
        delta_pi, empty_neighbors = utility_and_hypothetical_utility(x, neighbors, kappa_aa, kappa_bb, kappa_ab, kappa_ba)
        #p_accept = gamma / (1 + np.exp(-2/gamma * Delta_pi))
        #rates =  abs(x) * ((A * p_accept) @ (1 - abs(x)))

        rates, idx_map = compute_move_rates(delta_pi, gamma)
        total_rates = np.sum(rates)
        if total_rates == 0:
            break
        selected_move = np.random.choice(len(rates), p=rates/total_rates)
        node_i, vacant_k_th = idx_map[selected_move]
        move_to = empty_neighbors[node_i][vacant_k_th] 
        x[move_to] = x[node_i]
        x[node_i] = 0
        t += -np.log(np.random.rand()) / rates[selected_move]
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

if (kappa_ab*kappa_ba) < 0:
    #prepatterned condition
    y0 = make_wave_pattern(Lx, Ly,int(vacant*N), freq_x=4, freq_y=4, smoothness=3)

snapshots = simulation_with_snapshots(np.linspace(0, 100000, 1000000), y0, A, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba)

# --- Animation ---
fig, ax = plt.subplots()
cmap = plt.get_cmap('bwr', 3)  # blue-white-red
im = ax.imshow(snapshots[0], cmap=cmap, vmin=-1, vmax=1)
ax.set_title(fr"$D = {gamma/2},\, \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, b={b}, \, d={d} $")
plt.colorbar(im, ax=ax, ticks=[-1,0,1], label='State')

def update(frame):
    im.set_array(snapshots[frame])
    return [im]

anim = FuncAnimation(fig, update, frames=len(snapshots), interval=100, blit=True)
# --- Export as GIF ---
gif_writer = PillowWriter(fps=10)
anim.save("simulation.gif", writer=gif_writer)
plt.show()
