"""
2D implementation of sociohydrodynamic models based on Schelling's model and combined with the Voter model.

Model "Vitelli" incorporates a noisy version of the model presented in Seara et.al 2025. With local density phi_a 
of agents of type a evolving as:

∂t phi_a = \nabla  [D^a phi_0 \nabla phi_a - D^a phi_a \nabla  phi_0 + D^a \beta phi_a phi_0 \nabla  pi_a +  sqrt(2D^a h^2 phi_a phi_0) Z] 

with: 
phi_0 = 1 - sum_b phi_b (density of vacant sites) 
Z Gaussian white noise
pi_a = is the Gradient of a utility function U^a, such that:
pi_a = \nabla U^a = \nabla [ \sum_b \kappa^{ab} \phi_b + \sum_{b, c} \nu^{abc} \phi_b \phi_c + \sum_b \Gamma^{ab} \nabla^2 \phi_b ]
D^a is the type dependent Schelling diffusion constant
\beta is the inverse temperature controlling the strength of the utility gradient on the diffusion process
h is the microscopic lattice spacing (zero for the strict thermodynamic limit)

Model "Schelling" implements Schelling type rules where agents only diffuse if the fraction of like agents is below a threshold theta.
In that case the dynamical equations become:

∂t phi_a =   w0 (phi_0 ∂^2 phi_a - phi_a ∂^2 phi_0)  + phi_a * phi_0 * ∂^2 w0 + 2 phi_0 ∂ w0 . ∂ phi_a ) +  ∂[sqrt(2 w0 h^2 phi_a phi_0) Z] + R(phi)

with:
w0 = D/ (1+ e^(-\beta \pi)) density dependent diffusion term
pi = \sum_b ((theta - 1) delta^{ab} + theta sigma_x^{ab} ) (phi_b + \Gamma ∂^2 phi_b) utility threshold function with Gaussian smeared neighborhood (\sigma^2/2 = \Gamma)
Z Gaussian white noise
R(phi) implements a voter model with mean-field equation ∂t phi_a = phi*(b*pho_0 - d) with b and d birth and death rates respectively

Features:
- A class object fhd_2d for 2D (see FHD_1D.py for 1D version of the code)
- Positivity floor for densities
- multiplicative conservative noise ~ sqrt(phi_a phi_0), turn on by passing non-zero "toggle_noise"
- numerical differentiation by fft, derivatives are computed using finite differences when passing "fft = False".

Authors: Tuan Pham and Wout Merbis
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.fft import dct, idct
import scipy as sp

import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import numba as nb
from numba import njit, prange

from .operations import *

@njit(parallel=True, cache=True)
def _redistribute_low_species_local_numba(
    rho,
    species,
    projection_floor,
    projection_tol,
    radius,
    bc_periodic,
    n_sweeps,
):
    """
    Local conservative same-species repair for rho[species] < projection_floor.

    For each deficient cell, pull same-species mass from neighboring cells
    proportionally to their available same-species mass.

    Parallelization uses coloring so same-color cells have disjoint local
    neighborhoods and cannot race.

    Parameters
    ----------
    rho : array, shape (nspecies, Nx, Ny)
        Modified in-place.

    species : int
        Species index to repair.

    projection_floor : float
        Lower bound.

    radius : int
        Local redistribution radius. Usually 1.

    bc_periodic : bool
        Whether to wrap neighbor indices.

    n_sweeps : int
        Number of repeated local redistribution sweeps.

    Returns
    -------
    moved_total_all : float
        Total mass moved locally.

    leftover_total : float
        Remaining lower-bound deficit after local repair.
    """
    nspecies, Nx, Ny = rho.shape

    period = 2 * radius + 1
    moved_total_all = 0.0

    for _sweep in range(n_sweeps):
        for color_i in range(period):
            if color_i >= Nx:
                continue

            n_i_color = (Nx - 1 - color_i) // period + 1

            for color_j in range(period):
                if color_j >= Ny:
                    continue

                n_j_color = (Ny - 1 - color_j) // period + 1
                n_cells_color = n_i_color * n_j_color

                moved_color = 0.0

                for idx in prange(n_cells_color):
                    qi = idx // n_j_color
                    qj = idx - qi * n_j_color

                    i = color_i + qi * period
                    j = color_j + qj * period

                    current = rho[species, i, j]

                    if current >= projection_floor - projection_tol:
                        continue

                    deficit = projection_floor - current

                    # First pass: compute total available same-species mass
                    # in the local neighborhood.
                    total_available = 0.0

                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if di == 0 and dj == 0:
                                continue

                            ii = i + di
                            jj = j + dj

                            if bc_periodic:
                                if ii < 0:
                                    ii += Nx
                                elif ii >= Nx:
                                    ii -= Nx

                                if jj < 0:
                                    jj += Ny
                                elif jj >= Ny:
                                    jj -= Ny
                            else:
                                if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                                    continue

                            available = rho[species, ii, jj] - projection_floor

                            if available > 0.0:
                                total_available += available

                    if total_available <= 0.0:
                        continue

                    moved = deficit
                    if moved > total_available:
                        moved = total_available

                    # Second pass: remove from neighbors proportionally to
                    # local available mass.
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if di == 0 and dj == 0:
                                continue

                            ii = i + di
                            jj = j + dj

                            if bc_periodic:
                                if ii < 0:
                                    ii += Nx
                                elif ii >= Nx:
                                    ii -= Nx

                                if jj < 0:
                                    jj += Ny
                                elif jj >= Ny:
                                    jj -= Ny
                            else:
                                if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                                    continue

                            available = rho[species, ii, jj] - projection_floor

                            if available > 0.0:
                                take = moved * available / total_available
                                rho[species, ii, jj] -= take

                    rho[species, i, j] += moved
                    moved_color += moved

                moved_total_all += moved_color

    # Compute remaining deficit.
    leftover_total = 0.0

    for idx in prange(Nx * Ny):
        i = idx // Ny
        j = idx - i * Ny

        deficit = projection_floor - rho[species, i, j]
        if deficit > projection_tol:
            leftover_total += deficit

    return moved_total_all, leftover_total

@njit(parallel=True, cache=True)
def _redistribute_simplex_local_numba(
    rho,
    radius,
    bc_periodic,
    n_sweeps,
    projection_tol,
):
    """
    Local conservative repair for sum_a rho_a > 1.

    Overfull cells donate occupied density to neighboring cells with vacancy
    capacity. Donation is weighted by neighboring vacancy capacity and
    preserves the donor cell's species composition.

    Parallelization uses coloring so same-color cells have disjoint local
    neighborhoods and cannot race.

    Parameters
    ----------
    rho : array, shape (nspecies, Nx, Ny)
        Modified in-place.

    radius : int
        Local redistribution radius. Usually 1.

    bc_periodic : bool
        Whether to wrap neighbor indices.

    n_sweeps : int
        Number of repeated local redistribution sweeps.

    Returns
    -------
    moved_total_all : float
        Total occupied mass moved locally.

    leftover_total : float
        Remaining simplex excess after local repair.
    """
    nspecies, Nx, Ny = rho.shape

    period = 2 * radius + 1
    moved_total_all = 0.0

    for _sweep in range(n_sweeps):
        for color_i in range(period):
            if color_i >= Nx:
                continue

            n_i_color = (Nx - 1 - color_i) // period + 1

            for color_j in range(period):
                if color_j >= Ny:
                    continue

                n_j_color = (Ny - 1 - color_j) // period + 1
                n_cells_color = n_i_color * n_j_color

                moved_color = 0.0

                for idx in prange(n_cells_color):
                    qi = idx // n_j_color
                    qj = idx - qi * n_j_color

                    i = color_i + qi * period
                    j = color_j + qj * period

                    local_sum = 0.0
                    for a in range(nspecies):
                        local_sum += rho[a, i, j]

                    if local_sum <= 1.0 + projection_tol:
                        continue

                    excess = local_sum - 1.0

                    # First pass: compute total vacancy capacity in local
                    # neighborhood.
                    total_capacity = 0.0

                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if di == 0 and dj == 0:
                                continue

                            ii = i + di
                            jj = j + dj

                            if bc_periodic:
                                if ii < 0:
                                    ii += Nx
                                elif ii >= Nx:
                                    ii -= Nx

                                if jj < 0:
                                    jj += Ny
                                elif jj >= Ny:
                                    jj -= Ny
                            else:
                                if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                                    continue

                            neigh_sum = 0.0
                            for a in range(nspecies):
                                neigh_sum += rho[a, ii, jj]

                            capacity = 1.0 - neigh_sum

                            if capacity > 0.0:
                                total_capacity += capacity

                    if total_capacity <= 0.0:
                        continue

                    moved = excess
                    if moved > total_capacity:
                        moved = total_capacity

                    # Second pass: add to neighbors according to their
                    # vacancy capacity. Use source composition.
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if di == 0 and dj == 0:
                                continue

                            ii = i + di
                            jj = j + dj

                            if bc_periodic:
                                if ii < 0:
                                    ii += Nx
                                elif ii >= Nx:
                                    ii -= Nx

                                if jj < 0:
                                    jj += Ny
                                elif jj >= Ny:
                                    jj -= Ny
                            else:
                                if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                                    continue

                            neigh_sum = 0.0
                            for a in range(nspecies):
                                neigh_sum += rho[a, ii, jj]

                            capacity = 1.0 - neigh_sum

                            if capacity > 0.0:
                                neighbor_weight = capacity / total_capacity
                                total_to_neighbor = moved * neighbor_weight

                                for a in range(nspecies):
                                    frac_a = rho[a, i, j] / local_sum
                                    rho[a, ii, jj] += total_to_neighbor * frac_a

                    # Remove from source after adding, so source composition
                    # above remains the old pre-removal composition.
                    for a in range(nspecies):
                        frac_a = rho[a, i, j] / local_sum
                        rho[a, i, j] -= moved * frac_a

                    moved_color += moved

                moved_total_all += moved_color

    # Compute remaining simplex excess.
    leftover_total = 0.0

    for idx in prange(Nx * Ny):
        i = idx // Ny
        j = idx - i * Ny

        local_sum = 0.0
        for a in range(nspecies):
            local_sum += rho[a, i, j]

        excess = local_sum - 1.0
        if excess > projection_tol:
            leftover_total += excess

    return moved_total_all, leftover_total


class fhd_2d:
    '''Defines the 2-D fluctuating hydrodynamics class for simulating the sociohydrodynamic equations including noise and reactions:

    '''
    def __init__(self, L, N, 
                 bc = "periodic", 
                 fft = False, 
                 schelling_flux="finite_volume",
                 projection_floor=0.0,
                 projection_mode="redistribute",
                 projection_tol = 1e-14,
                 redistribute_radius=1,
                 redistribute_fallback="global",
                 use_numba_projection=False,
                 numba_projection_threads=4,):
        '''
        Initializes instance of the fhd class object

        Args:
            L:   tuple (Lx, Ly): spatial lengths of the domain, coordinates will be defined as running from -L/2 to L/2
            N:   tuple (Nx, Ny): number of discretization steps per coordinate
            bc:  boundary conditions, choose "periodic" or "Neumann"
            fft: Bool: when True derivatives are computed using FFT (only compatible with periodic bc's)
            schelling_flux: string: "collocated" for using 8-th order derivative stencils 
                            or "finite_volume" for using finite-volume implementation of the utility term
            projection_floor: float: densities below the projection_floor or above 1-projection_floor will be 
                            handled according to projection_mode
            projection_mode: string: select methodology for handling unphysical densities (simplex violations)
                "clip":              clips densities below projection_floor or above 1-projection_floor. Does not conserve species mass
                "transfer_to_other": transfers lower density violations to the other species type. 
                                     Upper bound violations are tranferred to vacant sites, so this method does not conserve species mass
                "redistribute":      Redistribute local simplex violations by pushing or pulling density to neighboring cells of the same type
                                     This method conserves mass density
            redistribute_radius: int: radius for neighboring cells used in "redistribute" projection mode
            redistribute_fallback: string: if no vacancy is available locally, redistribute projection_mode may fallback to
                "global":            redistribute globally across the entire lattice.
                "transfer_to_other": transfer species mass to other species.
        '''
        self.N = N
        self.L = L
        self.bc = bc
        self.fft = fft
        
        self.Lx, self.Ly = L
        self.Nx, self.Ny = N
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        
        if bc == "periodic":
            self.x = np.arange(-self.Lx/2, self.Lx/2, self.dx)
            self.y = np.arange(-self.Ly/2, self.Ly/2, self.dy)
        elif bc == "Neumann":
            self.N = (N[0]+1,N[1]+1)
            self.Nx += 1
            self.Ny += 1
            self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
            self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
            self.dx = self.Lx / self.Nx
            self.dy = self.Ly / self.Ny            
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'Neumann' or 'Dirichlet' ")
        
        self.schelling_flux = schelling_flux

        if self.schelling_flux not in ("collocated", "finite_volume"):
            raise ValueError(
                "schelling_flux should be 'collocated' or 'finite_volume'"
            )
        
        self.projection_floor = projection_floor
        self.projection_mode = projection_mode
        self.projection_tol = projection_tol
        self.projection_mass_tol = 1e-10   # accumulated leftover tolerance
        self.redistribute_radius = redistribute_radius
        self.redistribute_fallback = redistribute_fallback
        self.redistribute_n_iter = 1
        

        self.use_numba_projection = use_numba_projection
        self.numba_projection_threads = numba_projection_threads

        if self.use_numba_projection:
            nb.set_num_threads(self.numba_projection_threads)

        allowed_projection_modes = (
            "clip",
            "transfer_to_other",
            "redistribute",
        )

        if self.projection_mode not in allowed_projection_modes:
            raise ValueError(
                f"projection_mode must be one of {allowed_projection_modes}, "
                f"got {self.projection_mode}"
            )

        allowed_fallbacks = (
            "clip",
            "global",
            "transfer_to_other",
        )

        if self.redistribute_fallback not in allowed_fallbacks:
            raise ValueError(
                f"redistribute_fallback must be one of {allowed_fallbacks}, "
                f"got {self.redistribute_fallback}"
            )

        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)*2*np.pi
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)*2*np.pi
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        self.phi_floor = 1e-14
        self.nspecies = 2

        # Matrices Dx and Dy for 8-th order finite differences, needed for divergence function below
        self.Dx = makeD(self.Nx, self.dx, self.bc)
        self.Dy = makeD(self.Ny, self.dy, self.bc)
        if not fft:
            if bc == "Neumann" and schelling_flux=="finite_volume":
                self.D2x = makeD2_fv_neumann(self.Nx, self.dx)
                self.D2y = makeD2_fv_neumann(self.Ny, self.dy)
            else:
                self.D2x = makeD2(self.Nx, self.dx, bc=self.bc)
                self.D2y = makeD2(self.Ny, self.dy, bc=self.bc)

            self.D3x = makeD3(self.Nx, self.dx, self.bc)
            self.D3y = makeD3(self.Ny, self.dy, self.bc)

    def _init_projection_diagnostics(self):
        return {
            "n_calls": 0,

            # Lower clipping / repair
            "n_low_entries": 0,
            "mass_added_low": 0.0,
            "max_low_violation": 0.0,

            # Upper clipping / repair
            "n_high_entries": 0,
            "mass_removed_high": 0.0,
            "max_high_violation": 0.0,

            # Simplex projection / repair
            "n_simplex_cells": 0,
            "mass_removed_simplex": 0.0,
            "max_simplex_violation": 0.0,

            # Transfer-to-other diagnostics
            "mass_transferred_low": 0.0,
            "mass_transferred_AtoB": 0.0,
            "mass_transferred_BtoA": 0.0,
            "n_transfer_fallback_entries": 0,
            "mass_added_transfer_fallback": 0.0,
            "mass_transferred_to_vacancy": 0.0,
            "mass_transferred_A_to_vacancy": 0.0,
            "mass_transferred_B_to_vacancy": 0.0,
            "mass_transferred_high_to_vacancy": 0.0,
            "mass_transferred_simplex_to_vacancy": 0.0,

            # Local redistribution diagnostics
            "mass_redistributed_low_local": 0.0,
            "mass_redistributed_low_global": 0.0,
            "mass_redistributed_high_local": 0.0,
            "mass_redistributed_high_global": 0.0,
            "mass_redistributed_simplex_local": 0.0,
            "mass_redistributed_simplex_global": 0.0,

            "redistribute_low_leftover": 0.0,
            "redistribute_high_leftover": 0.0,
            "redistribute_simplex_leftover": 0.0,

            "n_redistribute_low_failures": 0,
            "n_redistribute_high_failures": 0,
            "n_redistribute_simplex_failures": 0,

            # Total bookkeeping
            "n_expensive_projection_calls": 0,
            "n_roundoff_cleanup_calls": 0,
            "mass_roundoff_cleanup": 0.0,
            "mass_before_projection": 0.0,
            "mass_after_projection": 0.0,
            "net_mass_change_projection": 0.0,

            "history": [],
        }

    def _ensure_work(self, dtype=np.float64):
        shape2 = (self.nspecies,) + self.N
        shape_vec = (2, self.nspecies) + self.N

        needs_new = (
            not hasattr(self, "_work")
            or self._work["divJ"].dtype != dtype
        )

        if needs_new:
            self._work = {
                "phi0": np.empty(self.N, dtype=dtype),
                "lap_phi": np.empty(shape2, dtype=dtype),
                "lap_phi0": np.empty(self.N, dtype=dtype),
                "pi": np.empty(shape2, dtype=dtype),
                "dUdx": np.empty(shape_vec, dtype=dtype),
                "flux": np.empty(shape_vec, dtype=dtype),
                "div_dUdx": np.empty(shape2, dtype=dtype),
                "divJ": np.empty(shape2, dtype=dtype),
                "voter_current": np.empty(self.N, dtype=dtype),

                # Derivative temporaries 
                "grad_pi": np.empty(shape_vec, dtype=dtype),
                "grad_lap_phi": np.empty(shape_vec, dtype=dtype),

                # Center mobility rho_a * rho_0
                "rho_center": np.empty(shape2, dtype=dtype),

                # Conservative face noise work arrays
                "dnoise_dx": np.empty(shape2, dtype=dtype),
                "tmp": np.empty(shape2, dtype=dtype),

                # Demographic voter noise
                "xi2": np.empty(self.N, dtype=dtype),
                "demo_noise": np.empty(self.N, dtype=dtype),
                "rho_ab": np.empty(self.N, dtype=dtype),

                # Integration step
                "phi_next": np.empty(shape2, dtype=dtype),
            }

            if self.bc == "periodic":
                face_x_shape = shape2
                face_y_shape = shape2
                det_face_x_shape = shape2
                det_face_y_shape = shape2
            elif self.bc == "Neumann":
                Nx, Ny = self.N
                face_x_shape = (self.nspecies, Nx + 1, Ny)
                face_y_shape = (self.nspecies, Nx, Ny + 1)
                det_face_x_shape = (self.nspecies, Nx + 1, Ny)
                det_face_y_shape = (self.nspecies, Nx, Ny + 1)
            else:
                raise ValueError(
                    f"Face-noise work arrays not implemented for bc={self.bc}"
                )

            self._work.update({
                "noise_flux_x": np.empty(face_x_shape, dtype=dtype),
                "noise_flux_y": np.empty(face_y_shape, dtype=dtype),
                "noise_amp_x": np.empty(face_x_shape, dtype=dtype),
                "noise_amp_y": np.empty(face_y_shape, dtype=dtype),
                "mobility": np.empty(shape2, dtype=dtype),
                "U": np.empty(shape2, dtype=dtype),
                "det_flux_x": np.empty(det_face_x_shape, dtype=dtype),
                "det_flux_y": np.empty(det_face_y_shape, dtype=dtype),
            })

            if "projection_diag" not in self._work:
                self._work["projection_diag"] = self._init_projection_diagnostics()

        return self._work
        
    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _shift2d(self, arr, di, dj, fill_value=0.0):
        """
        Shift a 2D array by (di, dj).

        For periodic boundaries, wrap.
        For Neumann/nonperiodic redistribution diagnostics, values shifted
        from outside are filled with fill_value.
        """
        if self.bc == "periodic":
            return np.roll(np.roll(arr, di, axis=0), dj, axis=1)

        out = np.full_like(arr, fill_value)

        Nx, Ny = arr.shape

        src_i0 = max(0, -di)
        src_i1 = min(Nx, Nx - di)
        dst_i0 = max(0, di)
        dst_i1 = min(Nx, Nx + di)

        src_j0 = max(0, -dj)
        src_j1 = min(Ny, Ny - dj)
        dst_j0 = max(0, dj)
        dst_j1 = min(Ny, Ny + dj)

        if src_i1 > src_i0 and src_j1 > src_j0:
            out[dst_i0:dst_i1, dst_j0:dst_j1] = arr[src_i0:src_i1, src_j0:src_j1]

        return out
    
    def _redistribution_offsets(self, radius=1, include_center=False):
        offsets = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if not include_center and di == 0 and dj == 0:
                    continue
                offsets.append((di, dj))
        return offsets

    def _neighbor_indices(self, i, j, radius=1, include_center=False):
        """
        Return local neighbor indices around (i, j).

        For periodic boundaries, neighbors wrap.
        For Neumann boundaries, out-of-domain neighbors are skipped.
        """
        Nx, Ny = self.N
        inds = []

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if not include_center and di == 0 and dj == 0:
                    continue

                ii = i + di
                jj = j + dj

                if self.bc == "periodic":
                    ii %= Nx
                    jj %= Ny
                else:
                    if ii < 0 or ii >= Nx or jj < 0 or jj >= Ny:
                        continue

                inds.append((ii, jj))

        return inds
    
    def _pull_species_mass_from_neighbors(self, rho, a, i, j, amount, 
                                          projection_floor=0.0, 
                                          radius=1,):
        """
        Increase rho[a, i, j] by taking the same species from nearby cells.

        This preserves species mass if enough local same-species mass is available.

        Returns
        -------
        moved : float
            Amount successfully moved locally.
        leftover : float
            Amount not moved.
        """
        if amount <= 0.0:
            return 0.0, 0.0

        neigh = self._neighbor_indices(i, j, radius=radius, include_center=False)

        if len(neigh) == 0:
            return 0.0, amount

        available = np.array(
            [
                max(rho[a, ii, jj] - projection_floor, 0.0)
                for ii, jj in neigh
            ],
            dtype=float,
        )

        total_available = float(available.sum())

        if total_available <= 0.0:
            return 0.0, amount

        moved = min(amount, total_available)
        weights = available / total_available

        for w, (ii, jj) in zip(weights, neigh):
            rho[a, ii, jj] -= moved * w

        rho[a, i, j] += moved

        return moved, amount - moved
    
    def _redistribute_low_species_weighted(
        self,
        rho,
        a,
        projection_floor=0.0,
        radius=1,
        n_iter=3,
        eps=1e-300,
    ):
        """
        Repair rho[a] < projection_floor by local same-species redistribution.

        This is symmetric over neighbor directions. Deficient cells receive mass
        from nearby donor cells proportionally to donor capacity.

        The transport is capped so that:
            - receivers do not receive more than their deficit;
            - donors do not donate more than their available mass.

        Returns
        -------
        moved_total : float
        leftover_total : float
        """
        u = rho[a]
        offsets = self._redistribution_offsets(radius=radius, include_center=False)

        # Uniform Moore-neighborhood weights by default.
        # You can replace this by distance weights if desired.
        weights = {
            (di, dj): 1.0
            for di, dj in offsets
        }

        moved_total_all = 0.0

        for _ in range(n_iter):
            deficit = np.maximum(projection_floor - u, 0.0)
            total_deficit = float(deficit.sum())

            if total_deficit <= 0.0:
                return moved_total_all, 0.0

            capacity = np.maximum(u - projection_floor, 0.0)

            # ------------------------------------------------------------
            # Receiver-side proposal:
            #
            # For each deficient receiver cell, distribute its deficit among
            # neighboring donors in proportion to donor capacity.
            # ------------------------------------------------------------
            denom_receiver = np.zeros_like(u)

            for di, dj in offsets:
                # donor at p sends to receiver p + (di,dj)
                donor_capacity_at_receiver = self._shift2d(
                    capacity,
                    di,
                    dj,
                    fill_value=0.0,
                )
                denom_receiver += weights[(di, dj)] * donor_capacity_at_receiver

            proposed_to_receiver = {}
            donor_out = np.zeros_like(u)

            active_receiver = denom_receiver > eps

            for di, dj in offsets:
                donor_capacity_at_receiver = self._shift2d(
                    capacity,
                    di,
                    dj,
                    fill_value=0.0,
                )

                proposal = np.zeros_like(u)
                proposal[active_receiver] = (
                    deficit[active_receiver]
                    * weights[(di, dj)]
                    * donor_capacity_at_receiver[active_receiver]
                    / denom_receiver[active_receiver]
                )

                # proposal is indexed at receiver cells.
                proposed_to_receiver[(di, dj)] = proposal

                # Shift back to donor coordinates to compute total proposed
                # outgoing mass from each donor.
                donor_out += self._shift2d(
                    proposal,
                    -di,
                    -dj,
                    fill_value=0.0,
                )

            # ------------------------------------------------------------
            # Donor-side cap:
            #
            # If many receivers ask from the same donor, scale all outgoing
            # proposals from that donor so it does not give more than capacity.
            # ------------------------------------------------------------
            donor_scale = np.ones_like(u)
            active_donor = donor_out > eps
            donor_scale[active_donor] = np.minimum(
                1.0,
                capacity[active_donor] / donor_out[active_donor],
            )

            received_total = np.zeros_like(u)
            donated_total = np.zeros_like(u)

            for di, dj in offsets:
                scale_at_receiver = self._shift2d(
                    donor_scale,
                    di,
                    dj,
                    fill_value=0.0,
                )

                actual_to_receiver = proposed_to_receiver[(di, dj)] * scale_at_receiver

                received_total += actual_to_receiver

                donated_total += self._shift2d(
                    actual_to_receiver,
                    -di,
                    -dj,
                    fill_value=0.0,
                )

            moved = float(received_total.sum())

            if moved <= 0.0:
                break

            u += received_total
            u -= donated_total

            moved_total_all += moved

        leftover = float(np.maximum(projection_floor - u, 0.0).sum())
        return moved_total_all, leftover

    def _redistribute_simplex_weighted(
        self,
        rho,
        radius=1,
        n_iter=3,
        eps=1e-300,
    ):
        """
        Repair sum_a rho_a > 1 by local conservative redistribution.

        Overfull cells donate occupied density to nearby cells with vacancy
        capacity. Transport is symmetric and capacity-weighted.

        Species composition of the donated mass follows the donor cell.

        Returns
        -------
        moved_total : float
        leftover_total : float
        """
        nspecies = rho.shape[0]
        offsets = self._redistribution_offsets(radius=radius, include_center=False)

        weights = {
            (di, dj): 1.0
            for di, dj in offsets
        }

        moved_total_all = 0.0

        for _ in range(n_iter):
            sumrho = rho.sum(axis=0)

            excess = np.maximum(sumrho - 1.0, 0.0)
            total_excess = float(excess.sum())

            if total_excess <= 0.0:
                return moved_total_all, 0.0

            capacity = np.maximum(1.0 - sumrho, 0.0)

            # ------------------------------------------------------------
            # Donor-side proposal:
            #
            # Each overfull donor distributes its excess among neighboring
            # receiver cells in proportion to their vacancy capacity.
            # ------------------------------------------------------------
            denom_donor = np.zeros_like(sumrho)

            for di, dj in offsets:
                receiver_capacity_at_donor = self._shift2d(
                    capacity,
                    -di,
                    -dj,
                    fill_value=0.0,
                )
                denom_donor += weights[(di, dj)] * receiver_capacity_at_donor

            proposed_from_donor = {}
            receiver_in = np.zeros_like(sumrho)

            active_donor = denom_donor > eps

            for di, dj in offsets:
                receiver_capacity_at_donor = self._shift2d(
                    capacity,
                    -di,
                    -dj,
                    fill_value=0.0,
                )

                proposal = np.zeros_like(sumrho)
                proposal[active_donor] = (
                    excess[active_donor]
                    * weights[(di, dj)]
                    * receiver_capacity_at_donor[active_donor]
                    / denom_donor[active_donor]
                )

                # proposal is indexed at donor cells.
                proposed_from_donor[(di, dj)] = proposal

                # Shift to receiver coordinates to compute total proposed incoming.
                receiver_in += self._shift2d(
                    proposal,
                    di,
                    dj,
                    fill_value=0.0,
                )

            # ------------------------------------------------------------
            # Receiver-side cap:
            #
            # If many donors send to the same receiver, scale incoming mass so
            # the receiver does not exceed its vacancy capacity.
            # ------------------------------------------------------------
            receiver_scale = np.ones_like(sumrho)
            active_receiver = receiver_in > eps
            receiver_scale[active_receiver] = np.minimum(
                1.0,
                capacity[active_receiver] / receiver_in[active_receiver],
            )

            # Species fractions at donor cells.
            species_frac = np.zeros_like(rho)
            valid = sumrho > eps

            for a in range(nspecies):
                species_frac[a, valid] = rho[a, valid] / sumrho[valid]

            removed_species = np.zeros_like(rho)
            added_species = np.zeros_like(rho)

            for di, dj in offsets:
                scale_at_donor = self._shift2d(
                    receiver_scale,
                    -di,
                    -dj,
                    fill_value=0.0,
                )

                actual_from_donor_total = (
                    proposed_from_donor[(di, dj)] * scale_at_donor
                )

                for a in range(nspecies):
                    actual_from_donor_a = (
                        actual_from_donor_total * species_frac[a]
                    )

                    removed_species[a] += actual_from_donor_a

                    added_species[a] += self._shift2d(
                        actual_from_donor_a,
                        di,
                        dj,
                        fill_value=0.0,
                    )

            moved = float(removed_species.sum())

            if moved <= 0.0:
                break

            rho -= removed_species
            rho += added_species

            moved_total_all += moved

        leftover = float(np.maximum(rho.sum(axis=0) - 1.0, 0.0).sum())
        return moved_total_all, leftover

    def _pull_species_mass_global(self, rho, a, i, j, amount, projection_floor=0.0):
        """
        Increase rho[a, i, j] by taking same-species mass globally.

        Nonlocal fallback, but species-mass conserving.
        """
        if amount <= 0.0:
            return 0.0, 0.0

        available = rho[a] - projection_floor
        available = np.maximum(available, 0.0)

        # Do not take from the target cell.
        available[i, j] = 0.0

        total_available = float(available.sum())

        if total_available <= 0.0:
            return 0.0, amount

        moved = min(amount, total_available)
        weights = available / total_available

        rho[a] -= moved * weights
        rho[a, i, j] += moved

        return moved, amount - moved

    def _push_species_to_vacancy_neighbors(self, rho, species_amounts, i, j, radius=1,):
        """
        Move species masses from cell (i, j) to nearby cells with vacancy capacity.

        Parameters
        ----------
        rho : array, shape (nspecies, Nx, Ny)

        species_amounts : array, shape (nspecies,)
            Amount of each species to move out of cell (i, j).

        Returns
        -------
        moved_total : float
        leftover_total : float
        """
        species_amounts = np.asarray(species_amounts, dtype=float)
        total_amount = float(species_amounts.sum())

        if total_amount <= 0.0:
            return 0.0, 0.0

        neigh = self._neighbor_indices(i, j, radius=radius, include_center=False)

        if len(neigh) == 0:
            return 0.0, total_amount

        capacities = np.array(
            [
                max(1.0 - float(rho[:, ii, jj].sum()), 0.0)
                for ii, jj in neigh
            ],
            dtype=float,
        )

        total_capacity = float(capacities.sum())

        if total_capacity <= 0.0:
            return 0.0, total_amount

        moved_total = min(total_amount, total_capacity)
        leftover_total = total_amount - moved_total

        neighbor_weights = capacities / total_capacity

        species_weights = species_amounts / total_amount
        moved_species = moved_total * species_weights

        # Remove from source cell.
        for a in range(rho.shape[0]):
            rho[a, i, j] -= moved_species[a]

        # Add to neighboring cells according to vacancy capacity.
        for w, (ii, jj) in zip(neighbor_weights, neigh):
            for a in range(rho.shape[0]):
                rho[a, ii, jj] += moved_species[a] * w

        return moved_total, leftover_total

    def _push_species_to_vacancy_global(self, rho, species_amounts, source_i=None, source_j=None,):
        """
        Move species masses to cells with vacancy capacity globally.

        Nonlocal fallback, but conserves species masses if enough capacity exists.
        """
        species_amounts = np.asarray(species_amounts, dtype=float)
        total_amount = float(species_amounts.sum())

        if total_amount <= 0.0:
            return 0.0, 0.0

        capacity = 1.0 - rho.sum(axis=0)
        capacity = np.maximum(capacity, 0.0)

        if source_i is not None and source_j is not None:
            capacity[source_i, source_j] = 0.0

        total_capacity = float(capacity.sum())

        if total_capacity <= 0.0:
            return 0.0, total_amount

        moved_total = min(total_amount, total_capacity)
        leftover_total = total_amount - moved_total

        weights = capacity / total_capacity
        species_weights = species_amounts / total_amount
        moved_species = moved_total * species_weights

        for a in range(rho.shape[0]):
            rho[a] += moved_species[a] * weights

        return moved_total, leftover_total

    def _repair_low_species_global_vectorized(
        self,
        rho,
        a,
        projection_floor=0.0,
    ):
        """
        Global same-species repair for any remaining negative values.

        Conserves species mass if enough global available mass exists.
        """
        u = rho[a]

        deficit = np.maximum(projection_floor - u, 0.0)
        total_deficit = float(deficit.sum())

        if total_deficit <= 0.0:
            return 0.0, 0.0

        available = np.maximum(u - projection_floor, 0.0)
        total_available = float(available.sum())

        moved = min(total_deficit, total_available)

        if moved <= 0.0:
            return 0.0, total_deficit

        # Fill deficient cells proportionally to deficit.
        u += moved * deficit / total_deficit

        # Remove from available cells proportionally to availability.
        u -= moved * available / total_available

        leftover = total_deficit - moved
        return moved, leftover

    def _repair_simplex_global_vectorized(self, rho, eps=1e-300):
        """
        Global conservative repair of simplex violations.

        Moves excess occupied density from overfull cells to cells with vacancy
        capacity. Preserves species masses if enough total capacity exists.
        """
        nspecies = rho.shape[0]
        sumrho = rho.sum(axis=0)

        excess = np.maximum(sumrho - 1.0, 0.0)
        total_excess = float(excess.sum())

        if total_excess <= 0.0:
            return 0.0, 0.0

        capacity = np.maximum(1.0 - sumrho, 0.0)
        total_capacity = float(capacity.sum())

        moved = min(total_excess, total_capacity)

        if moved <= 0.0:
            return 0.0, total_excess

        # Remove from overfull cells according to their local composition.
        source_weight = excess / total_excess
        target_weight = capacity / total_capacity

        species_removed = np.zeros(nspecies, dtype=float)

        mask = sumrho > eps
        for a in range(nspecies):
            frac_a = np.zeros_like(sumrho)
            frac_a[mask] = rho[a, mask] / sumrho[mask]

            remove_a = moved * source_weight * frac_a
            rho[a] -= remove_a
            species_removed[a] = float(remove_a.sum())

        # Add species masses to vacancy cells according to capacity.
        for a in range(nspecies):
            rho[a] += species_removed[a] * target_weight

        leftover = total_excess - moved
        return moved, leftover

    def _project_density_redistribute(self, rho, diag, projection_floor=0.0):
        """
        Conservative local redistribution projection.

        Repair order:
            1. rho_a < projection_floor
            2. sum_a rho_a > 1
            3. rho_a > 1 - projection_floor

        The main repairs are local and conservative. Any remaining unrepairable
        amount falls back according to self.redistribute_fallback.
        """
        nspecies, Nx, Ny = rho.shape

        radius = self.redistribute_radius
        tol = getattr(self, "projection_tol", 100.0 * np.finfo(rho.dtype).eps)
        upper_bound = 1.0 - projection_floor

        bc_periodic = self.bc == "periodic"

        n_iter_default = getattr(self, "redistribute_n_iter", 1)
        n_iter_low = getattr(self, "redistribute_low_n_iter", n_iter_default)
        n_iter_simplex = getattr(
            self,
            "redistribute_simplex_n_iter",
            n_iter_default,
        )
        mass_tol = getattr(self, "projection_mass_tol", 1000.0 * np.finfo(rho.dtype).eps * rho.size,)

        def add_float(key, value):
            diag[key] = diag.get(key, 0.0) + float(value)

        def add_int(key, value):
            diag[key] = diag.get(key, 0) + int(value)

        # ------------------------------------------------------------
        # 1. Repair lower violations: rho_a < projection_floor.
        #    Move same-species mass from nearby cells into deficient cells.
        # ------------------------------------------------------------
        # low_mask = rho < projection_floor

        for a in range(nspecies):
            low_mask_a = rho[a] < projection_floor - tol

            if not np.any(low_mask_a):
                continue

            if self.use_numba_projection:
                moved_local, leftover = _redistribute_low_species_local_numba(
                    rho,
                    a,
                    projection_floor,
                    self.projection_tol,
                    radius,
                    bc_periodic,
                    n_iter_low,
                )
            else:
                moved_local, leftover = self._redistribute_low_species_weighted(
                    rho,
                    a,
                    projection_floor=projection_floor,
                    radius=radius,
                    n_iter=n_iter_low,
                )

            add_float("mass_redistributed_low_local", moved_local)

            if leftover > mass_tol and self.redistribute_fallback == "global":
                moved_global, leftover = self._repair_low_species_global_vectorized(
                    rho,
                    a,
                    projection_floor=projection_floor,
                )
                add_float("mass_redistributed_low_global", moved_global)

        # Optional fallback: after local redistribution, repair any remaining
        # lower violations by A <-> B transfer.
        if self.redistribute_fallback == "transfer_to_other":
            remaining_low = rho < projection_floor - tol

            if np.any(remaining_low):
                if nspecies != 2:
                    raise NotImplementedError(
                        "redistribute_fallback='transfer_to_other' is implemented "
                        "only for two species."
                    )

                (
                    transferred,
                    fallback_added,
                    n_fallback,
                    transferred_B_to_A,
                    transferred_A_to_B,
                ) = self._project_lower_transfer_to_other_2species(
                    rho,
                    projection_floor=projection_floor,
                )

                add_float("mass_transferred_low", transferred)
                add_float("mass_transferred_B_to_A", transferred_B_to_A)
                add_float("mass_transferred_A_to_B", transferred_A_to_B)
                add_int("n_transfer_fallback_entries", n_fallback)
                add_float("mass_added_transfer_fallback", fallback_added)

                # Only the fallback clipping part changes total mass.
                add_float("mass_added_low", fallback_added)

        # Final lower fallback: nonconservative clipping.
        remaining_low = rho <  projection_floor - tol
        leftover_mass = 0.0

        if np.any(remaining_low):
            leftover_mass = float((projection_floor - rho[remaining_low]).sum())

        if leftover_mass > mass_tol:
            n_fail = int(np.count_nonzero(remaining_low))

            rho[remaining_low] = projection_floor

            diag["mass_added_low"] += leftover_mass
            diag["redistribute_low_leftover"] += leftover_mass
            diag["n_redistribute_low_failures"] += n_fail
        else:
            # Roundoff cleanup only; do not count as a physical failure.
            tiny_low = (rho < projection_floor) & (rho >= projection_floor - tol)
            if np.any(tiny_low):
                rho[tiny_low] = projection_floor

        # ------------------------------------------------------------
        # 2. Repair simplex violations: sum_a rho_a > 1.
        #    Move occupied mass to nearby vacancy capacity while preserving
        #    source-cell species composition.
        #
        #    Once all rho_a >= 0, this also handles rho_a > 1 in practice.
        # ------------------------------------------------------------

        sumrho = rho.sum(axis=0)
        simplex_mask = sumrho > 1.0 + tol

        if np.any(simplex_mask):
            if self.use_numba_projection:
                moved_simplex_local, leftover = _redistribute_simplex_local_numba(
                    rho,
                    radius,
                    bc_periodic,
                    n_iter_simplex,
                    tol
                )
            else:
                moved_simplex_local, leftover = self._redistribute_simplex_weighted(
                    rho,
                    radius=radius,
                    n_iter=n_iter_simplex,
                )

            diag["mass_redistributed_simplex_local"] += moved_simplex_local

            if leftover > mass_tol and self.redistribute_fallback == "global":
                moved_global, leftover = self._repair_simplex_global_vectorized(rho)
                diag["mass_redistributed_simplex_global"] += moved_global

            if leftover > mass_tol:
                sumrho = rho.sum(axis=0)
                simplex_mask = sumrho > 1.0

                if np.any(simplex_mask):
                    removed = float((sumrho[simplex_mask] - 1.0).sum())
                    rho[:, simplex_mask] /= sumrho[simplex_mask][np.newaxis, :]

                    diag["mass_removed_simplex"] += removed
                    diag["redistribute_simplex_leftover"] += removed
                    diag["n_redistribute_simplex_failures"] += int(
                        np.count_nonzero(simplex_mask)
                    )

        # ------------------------------------------------------------
        # 3. Final upper-bound check: rho_a > 1 - projection_floor.
        #
        #    This should rarely trigger, because after lower repair and simplex
        #    repair, nonnegative densities with sum_a rho_a <= 1 automatically
        #    satisfy rho_a <= 1.
        # ------------------------------------------------------------
        # high_mask = rho > upper_bound
        high_mask = rho > upper_bound + tol

        if np.any(high_mask):
            n_high_this = int(np.count_nonzero(high_mask))
            max_high_violation_this = float(
                np.max(rho[high_mask] - upper_bound)
            )

            add_int("n_high_entries", n_high_this)
            diag["max_high_violation"] = max(
                diag.get("max_high_violation", 0.0),
                max_high_violation_this,
            )

            high_positions = np.argwhere(high_mask)

            for a, i, j in high_positions:
                current = rho[a, i, j]

                if current <= upper_bound:
                    continue

                excess = float(current - upper_bound)

                species_amounts = np.zeros(nspecies, dtype=rho.dtype)
                species_amounts[a] = excess

                moved_local, leftover = self._push_species_to_vacancy_neighbors(
                    rho,
                    species_amounts,
                    i,
                    j,
                    radius=radius,
                )

                add_float("mass_redistributed_high_local", moved_local)

                if leftover > 0.0 and self.redistribute_fallback == "global":
                    take = min(
                        leftover,
                        max(float(rho[a, i, j] - upper_bound), 0.0),
                    )

                    if take > 0.0:
                        rho[a, i, j] -= take

                        species_amounts2 = np.zeros(nspecies, dtype=rho.dtype)
                        species_amounts2[a] = take

                        moved_global, global_leftover = (
                            self._push_species_to_vacancy_global(
                                rho,
                                species_amounts2,
                                source_i=i,
                                source_j=j,
                            )
                        )

                        add_float("mass_redistributed_high_global", moved_global)
                        leftover = global_leftover

                # Final upper fallback: remove excess mass.
                if leftover > 0.0:
                    removable = max(float(rho[a, i, j] - upper_bound), 0.0)
                    removed = min(leftover, removable)

                    if removed > 0.0:
                        rho[a, i, j] -= removed

                        add_float("mass_removed_high", removed)
                        add_float("redistribute_high_leftover", removed)
                        add_int("n_redistribute_high_failures", 1)

        # ------------------------------------------------------------
        # 4. Tiny roundoff cleanup only.
        # ------------------------------------------------------------
        tiny_low = (rho < projection_floor) & (rho > projection_floor - tol)
        if np.any(tiny_low):
            rho[tiny_low] = projection_floor

        sumrho = rho.sum(axis=0)
        tiny_simplex = (sumrho > 1.0) & (sumrho < 1.0 + tol)
        if np.any(tiny_simplex):
            rho[:, tiny_simplex] /= sumrho[tiny_simplex][np.newaxis, :]

        tiny_high = (rho > upper_bound) & (rho < upper_bound + tol)
        if np.any(tiny_high):
            rho[tiny_high] = upper_bound

    def project_density_fast(self, rho):
        """
        Fast nonconservative projection to the physical simplex.

        This is used both for ordinary clipping and as a cheap roundoff cleanup
        when no real projection violation is present.
        """
        projection_floor = getattr(self, "projection_floor", 0.0)

        np.maximum(rho, projection_floor, out=rho)
        np.minimum(rho, 1.0 - projection_floor, out=rho)

        sumrho = rho[0] + rho[1]
        mask = sumrho > 1.0

        if np.any(mask):
            rho[0, mask] /= sumrho[mask]
            rho[1, mask] /= sumrho[mask]

        return rho
    
    def _project_lower_transfer_to_other_2species(
        self,
        rho,
        projection_floor=0.0,
    ):
        """
        Returns
        -------
        transferred_mass : float
            Amount of negative deficit compensated by the other species.

        fallback_added_mass : float
            Mass that still had to be added because local compensation failed.

        n_fallback_entries : int
            Number of entries still below projection_floor after local transfer.
        """
        A = rho[0]
        B = rho[1]

        transferred_mass = 0.0
        transferred_A_to_B = 0.0
        transferred_B_to_A = 0.0

        # Repair A by taking from B: B -> A
        mask_A = A < projection_floor
        if np.any(mask_A):
            deficit_A = projection_floor - A[mask_A]
            transferred_B_to_A = float(deficit_A.sum())
            A[mask_A] = projection_floor
            B[mask_A] -= deficit_A
            transferred_mass += transferred_B_to_A

        # Repair B by taking from A: A -> B
        mask_B = B < projection_floor
        if np.any(mask_B):
            deficit_B = projection_floor - B[mask_B]
            transferred_A_to_B = float(deficit_B.sum())
            B[mask_B] = projection_floor
            A[mask_B] -= deficit_B
            transferred_mass += transferred_A_to_B

        # Fallback if compensation made the other species negative.
        fallback_added_mass = 0.0
        n_fallback_entries = 0

        mask_A_again = A < projection_floor
        if np.any(mask_A_again):
            old = float(A[mask_A_again].sum())
            n = int(np.count_nonzero(mask_A_again))
            added = n * projection_floor - old
            fallback_added_mass += float(added)
            n_fallback_entries += n
            A[mask_A_again] = projection_floor

        mask_B_again = B < projection_floor
        if np.any(mask_B_again):
            old = float(B[mask_B_again].sum())
            n = int(np.count_nonzero(mask_B_again))
            added = n * projection_floor - old
            fallback_added_mass += float(added)
            n_fallback_entries += n
            B[mask_B_again] = projection_floor

        return transferred_mass, fallback_added_mass, n_fallback_entries, transferred_B_to_A, transferred_A_to_B
    
    def _project_high_transfer_to_vacancy(
        self,
        rho,
        projection_floor=0.0,
    ):
        """
        Repair rho_a > 1 - projection_floor by transferring excess occupied
        density to vacancy.

        This is local and simplex-compatible, but it removes occupied mass.
        """
        upper_bound = 1.0 - projection_floor
        nspecies = rho.shape[0]

        high_mask = rho > upper_bound

        removed_total = 0.0
        removed_species = np.zeros(nspecies, dtype=float)
        n_high = 0
        max_violation = 0.0

        if not np.any(high_mask):
            return removed_total, removed_species, n_high, max_violation

        n_high = int(np.count_nonzero(high_mask))
        max_violation = float(np.max(rho[high_mask] - upper_bound))

        for a in range(nspecies):
            mask_a = rho[a] > upper_bound

            if not np.any(mask_a):
                continue

            excess_a = rho[a, mask_a] - upper_bound
            removed_a = float(excess_a.sum())

            rho[a, mask_a] = upper_bound

            removed_species[a] += removed_a
            removed_total += removed_a

        return removed_total, removed_species, n_high, max_violation

    def _project_simplex_transfer_to_vacancy(
        self,
        rho,
        ):
        """
        Repair sum_a rho_a > 1 by transferring excess occupied density
        to vacancy, preserving local species composition.

        This is exactly local simplex rescaling:
            rho_a <- rho_a / sumrho

        but returns species-resolved removed masses.
        """
        nspecies = rho.shape[0]

        sumrho = rho.sum(axis=0)
        simplex_mask = sumrho > 1.0

        removed_total = 0.0
        removed_species = np.zeros(nspecies, dtype=float)
        n_simplex = 0
        max_violation = 0.0

        if not np.any(simplex_mask):
            return removed_total, removed_species, n_simplex, max_violation

        n_simplex = int(np.count_nonzero(simplex_mask))
        excess = sumrho[simplex_mask] - 1.0

        removed_total = float(excess.sum())
        max_violation = float(excess.max())

        old_species_mass = rho[:, simplex_mask].sum(axis=1).copy()

        rho[:, simplex_mask] /= sumrho[simplex_mask][np.newaxis, :]

        new_species_mass = rho[:, simplex_mask].sum(axis=1).copy()
        removed_species[:] = old_species_mass - new_species_mass

        return removed_total, removed_species, n_simplex, max_violation

    def _project_density_transfer_to_other(
        self,
        rho,
        diag,
        projection_floor=0.0,
    ):
        """
        Project densities back to the local physical simplex for two species.

            rho_a >= projection_floor
            rho_a <= 1 - projection_floor
            sum_a rho_a <= 1

        Interpretation
        --------------
        Lower violations:
            transfer density between A and B.

        Upper and simplex violations:
            transfer occupied density to vacancy.

        This preserves local occupied density for lower violations, but removes
        occupied density for upper/simplex violations because vacancy is not an
        explicitly stored conserved field.
        """
        if rho.shape[0] != 2:
            raise NotImplementedError(
                "projection_mode='transfer_to_other' is currently implemented "
                "only for two species."
            )
        
        tol = self.projection_tol

        upper_bound = 1.0 - projection_floor + tol

        # ------------------------------------------------------------
        # 1. Lower repair: A < floor or B < floor.
        #    Transfer between A and B.
        # ------------------------------------------------------------
        low_mask = rho < projection_floor - tol

        if np.any(low_mask):
            n_low_this = int(np.count_nonzero(low_mask))
            max_low_violation_this = float(
                np.max(projection_floor - rho[low_mask])
            )

            transferred, fallback_added, n_fallback, transferred_B_to_A, transferred_A_to_B = (
                self._project_lower_transfer_to_other_2species(
                    rho,
                    projection_floor=projection_floor,
                )
            )

            diag["n_low_entries"] += n_low_this
            diag["max_low_violation"] = max(
                diag["max_low_violation"],
                max_low_violation_this,
            )

            diag["mass_transferred_low"] += transferred
            diag["mass_transferred_B_to_A"] += transferred_B_to_A
            diag["mass_transferred_A_to_B"] += transferred_A_to_B

            diag["n_transfer_fallback_entries"] += n_fallback
            diag["mass_added_transfer_fallback"] += fallback_added

            # Only fallback clipping actually adds occupied mass.
            diag["mass_added_low"] += fallback_added
                    
        # ------------------------------------------------------------
        # 2. Simplex repair: rho_A + rho_B > 1.
        #    Transfer occupied excess to vacancy, preserving composition.
        # ------------------------------------------------------------
        sumrho = rho.sum(axis=0)
        simplex_mask = sumrho > 1.0 + tol

        if np.any(simplex_mask):
            removed_simplex, removed_simplex_species, n_simplex_this, max_simplex_violation_this = (
                self._project_simplex_transfer_to_vacancy(rho)
            )

            diag["n_simplex_cells"] += n_simplex_this
            diag["mass_removed_simplex"] += removed_simplex
            diag["max_simplex_violation"] = max(
                diag["max_simplex_violation"],
                max_simplex_violation_this,
            )

            diag["mass_transferred_to_vacancy"] += removed_simplex
            diag["mass_transferred_simplex_to_vacancy"] += removed_simplex
            diag["mass_transferred_A_to_vacancy"] += removed_simplex_species[0]
            diag["mass_transferred_B_to_vacancy"] += removed_simplex_species[1]

        # ------------------------------------------------------------
        # 3. Upper repair: rho_a > 1 - floor.
        #    Transfer excess occupied species density to vacancy.
        # ------------------------------------------------------------
        high_mask = rho > upper_bound

        if np.any(high_mask):
            removed_high, removed_high_species, n_high_this, max_high_violation_this = (
                self._project_high_transfer_to_vacancy(
                    rho,
                    projection_floor=projection_floor,
                )
            )

            diag["n_high_entries"] += n_high_this
            diag["mass_removed_high"] += removed_high
            diag["max_high_violation"] = max(
                diag["max_high_violation"],
                max_high_violation_this,
            )

            diag["mass_transferred_to_vacancy"] += removed_high
            diag["mass_transferred_high_to_vacancy"] += removed_high
            diag["mass_transferred_A_to_vacancy"] += removed_high_species[0]
            diag["mass_transferred_B_to_vacancy"] += removed_high_species[1]

    def project_density(
        self,
        rho,
        work=None,
        step=None,
        record_history=False,
    ):
        """
        Project densities back to the physical simplex.

        Modes
        -----
        clip:
            Ordinary nonconservative clipping.

        transfer_to_other:
            Two-species local repair for negative densities by transferring
            mass between A and B.

        redistribute:
            Local species-conserving redistribution of negative, high, and
            simplex-violating densities.

        Adaptive behavior
        -----------------
        The expensive projection mode is called only if there is a real violation
        beyond self.projection_tol. Otherwise, only a cheap roundoff cleanup is
        applied using project_density_fast().
        """
        if work is None:
            work = self._work

        diag = work.get("projection_diag", None)
        if diag is None:
            work["projection_diag"] = self._init_projection_diagnostics()
            diag = work["projection_diag"]

        projection_floor = getattr(self, "projection_floor", 0.0)
        projection_tol = getattr(
            self,
            "projection_tol",
            100.0 * np.finfo(rho.dtype).eps,
        )
        upper_bound = 1.0 - projection_floor

        # ------------------------------------------------------------
        # Total mass bookkeeping is done only here.
        # ------------------------------------------------------------
        mass_before_all = float(rho.sum())

        diag["n_calls"] += 1
        diag["mass_before_projection"] += mass_before_all

        min_pre_projection = float(rho.min())
        max_pre_projection = float(rho.max())

        sumrho_pre = rho.sum(axis=0)
        max_sum_pre_projection = float(sumrho_pre.max())

        # ------------------------------------------------------------
        # Count only real violations, not roundoff-level violations.
        # ------------------------------------------------------------
        low_mask_pre = rho < projection_floor - projection_tol
        high_mask_pre = rho > upper_bound + projection_tol
        simplex_mask_pre = sumrho_pre > 1.0 + projection_tol

        n_low_this = int(np.count_nonzero(low_mask_pre))
        n_high_this = int(np.count_nonzero(high_mask_pre))
        n_simplex_this = int(np.count_nonzero(simplex_mask_pre))

        max_low_violation_this = 0.0
        max_high_violation_this = 0.0
        max_simplex_violation_this = 0.0

        if n_low_this > 0:
            max_low_violation_this = float(
                np.max(projection_floor - rho[low_mask_pre])
            )

        if n_high_this > 0:
            max_high_violation_this = float(
                np.max(rho[high_mask_pre] - upper_bound)
            )

        if n_simplex_this > 0:
            max_simplex_violation_this = float(
                np.max(sumrho_pre[simplex_mask_pre] - 1.0)
            )

        diag["n_low_entries"] += n_low_this
        diag["n_high_entries"] += n_high_this
        diag["n_simplex_cells"] += n_simplex_this

        diag["max_low_violation"] = max(
            diag["max_low_violation"],
            max_low_violation_this,
        )
        diag["max_high_violation"] = max(
            diag["max_high_violation"],
            max_high_violation_this,
        )
        diag["max_simplex_violation"] = max(
            diag["max_simplex_violation"],
            max_simplex_violation_this,
        )

        needs_projection = (
            n_low_this > 0
            or n_high_this > 0
            or n_simplex_this > 0
        )

        # Optional diagnostic keys. Safe even if not initialized explicitly.
        if not needs_projection:
            diag["n_roundoff_cleanup_calls"] = (
                diag.get("n_roundoff_cleanup_calls", 0) + 1
            )

        # Per-call mass changes for history.
        mass_added_low_before = diag["mass_added_low"]
        mass_removed_high_before = diag["mass_removed_high"]
        mass_removed_simplex_before = diag["mass_removed_simplex"]

        roundoff_mass_change_this = 0.0

        # ------------------------------------------------------------
        # Adaptive projection dispatch.
        # ------------------------------------------------------------
        if not needs_projection:
            # Only roundoff-level violations are present. Use the cheap clipping
            # projection but record its total mass effect separately from the real
            # projection diagnostics.
            diag["n_roundoff_cleanup_calls"] = (
                diag.get("n_roundoff_cleanup_calls", 0) + 1
            )
            mass_before_cleanup = float(rho.sum())
            self.project_density_fast(rho)
            mass_after_cleanup = float(rho.sum())

            roundoff_mass_change_this = mass_after_cleanup - mass_before_cleanup

            diag["mass_roundoff_cleanup"] = (
                diag.get("mass_roundoff_cleanup", 0.0)
                + roundoff_mass_change_this
            )

        elif self.projection_mode == "redistribute":
            diag["n_expensive_projection_calls"] = (
               diag.get("n_expensive_projection_calls", 0) + 1
            )

            self._project_density_redistribute(
                rho,
                diag,
                projection_floor=projection_floor,
            )

        elif self.projection_mode == "transfer_to_other":
            diag["n_expensive_projection_calls"] = (
                diag.get("n_expensive_projection_calls", 0) + 1
            )

            self._project_density_transfer_to_other(
                rho,
                diag,
                projection_floor=projection_floor,
            )

        elif self.projection_mode == "clip":
            # Use the existing fast nonconservative projection, but count the
            # corresponding mass changes explicitly for diagnostics.
            diag["n_expensive_projection_calls"] = (
                diag.get("n_expensive_projection_calls", 0) + 1
            )
            low_mask = rho < projection_floor
            if np.any(low_mask):
                old_low_mass = float(rho[low_mask].sum())
                n_low = int(np.count_nonzero(low_mask))
                added = float(n_low * projection_floor - old_low_mass)

                rho[low_mask] = projection_floor
                diag["mass_added_low"] += added

            sumrho = rho.sum(axis=0)
            simplex_mask = sumrho > 1.0

            if np.any(simplex_mask):
                excess = sumrho[simplex_mask] - 1.0
                removed = float(excess.sum())

                rho[:, simplex_mask] /= sumrho[simplex_mask][np.newaxis, :]
                diag["mass_removed_simplex"] += removed

            high_mask = rho > upper_bound
            if np.any(high_mask):
                old_high_mass = float(rho[high_mask].sum())
                n_high = int(np.count_nonzero(high_mask))
                removed = float(old_high_mass - n_high * upper_bound)

                rho[high_mask] = upper_bound
                diag["mass_removed_high"] += removed

        else:
            raise ValueError(f"Unknown projection_mode: {self.projection_mode}")

        # ------------------------------------------------------------
        # Final total mass bookkeeping.
        # ------------------------------------------------------------
        mass_after_all = float(rho.sum())
        net_change = mass_after_all - mass_before_all

        diag["mass_after_projection"] += mass_after_all
        diag["net_mass_change_projection"] += net_change

        mass_added_low_this = (
            diag["mass_added_low"] - mass_added_low_before
        )
        mass_removed_high_this = (
            diag["mass_removed_high"] - mass_removed_high_before
        )
        mass_removed_simplex_this = (
            diag["mass_removed_simplex"] - mass_removed_simplex_before
        )

        if record_history:
            diag["history"].append({
                "step": step,

                "needs_projection": needs_projection,

                "mass_before": mass_before_all,
                "mass_after": mass_after_all,
                "net_change": net_change,

                "n_low_entries": n_low_this,
                "n_high_entries": n_high_this,
                "n_simplex_cells": n_simplex_this,

                "mass_added_low": mass_added_low_this,
                "mass_removed_high": mass_removed_high_this,
                "mass_removed_simplex": mass_removed_simplex_this,
                "mass_roundoff_cleanup": roundoff_mass_change_this,

                "max_low_violation": max_low_violation_this,
                "max_high_violation": max_high_violation_this,
                "max_simplex_violation": max_simplex_violation_this,

                "min_pre_projection": min_pre_projection,
                "max_pre_projection": max_pre_projection,
                "max_sum_pre_projection": max_sum_pre_projection,

                "mass_redistributed_low_local": diag["mass_redistributed_low_local"],
                "mass_redistributed_low_global": diag["mass_redistributed_low_global"],
                "mass_redistributed_high_local": diag["mass_redistributed_high_local"],
                "mass_redistributed_high_global": diag["mass_redistributed_high_global"],
                "mass_redistributed_simplex_local": diag["mass_redistributed_simplex_local"],
                "mass_redistributed_simplex_global": diag["mass_redistributed_simplex_global"],
            })

        return rho

    def grad(self, u, out=None):
        if self.fft:
            u_hat = np.fft.fft2(u)

            grad_x_hat = 1j * self.kx * u_hat
            grad_y_hat = 1j * self.ky * u_hat

            if out is None:
                out = np.empty((2,) + u.shape, dtype=u.dtype)

            out[0] = np.fft.ifft2(grad_x_hat).real
            out[1] = np.fft.ifft2(grad_y_hat).real
            return out

        ush = u.shape

        if out is None:
            out = np.empty((2,) + ush, dtype=u.dtype)

        for i in range(ush[0]):
            out[0, i] = self.Dx.dot(u[i])
            out[1, i] = self.Dy.dot(u[i].T).T

        return out
    
    def lapl(self, u, out=None):
        if self.fft:
            u_hat = np.fft.fft2(u)

            lapl_hat = -(self.kx**2 + self.ky**2) * u_hat

            if out is None:
                out = np.empty_like(u)

            out[:] = np.fft.ifft2(lapl_hat).real
            return out

        ush = u.shape

        if out is None:
            out = np.empty_like(u)

        for i in range(ush[0]):
            out[i] = self.D2x.dot(u[i])
            out[i] += self.D2y.dot(u[i].T).T

        return out
    
    def grad_lapl(self, u, out=None):
        if self.fft:
            u_hat = np.fft.fft2(u)

            gradlap_x_hat = -1j * self.kx * (self.kx**2 + self.ky**2) * u_hat
            gradlap_y_hat = -1j * self.ky * (self.kx**2 + self.ky**2) * u_hat

            if out is None:
                out = np.empty((2,) + u.shape, dtype=u.dtype)

            out[0] = np.fft.ifft2(gradlap_x_hat).real
            out[1] = np.fft.ifft2(gradlap_y_hat).real
            return out

        ush = u.shape

        if out is None:
            out = np.empty((2,) + ush, dtype=u.dtype)

        for i in range(ush[0]):
            # x-component: D3x u + Dx D2y u
            out[0, i] = self.D3x.dot(u[i])
            out[0, i] += self.Dx.dot(self.D2y.dot(u[i].T).T)

            # y-component: D3y u + Dy D2x u
            out[1, i] = self.D3y.dot(u[i].T).T
            out[1, i] += self.Dy.dot(self.D2x.dot(u[i]).T).T

        return out
    
    def div2d(self, vec, out=None):
            """
            Divergence of a spatial vector field.

            vec shape: (2, nspecies, Nx, Ny)
            out shape: (nspecies, Nx, Ny)
            """
            nspecies = vec.shape[1]

            if out is None:
                out = np.empty(vec.shape[1:], dtype=vec.dtype)

            for i in range(nspecies):
                out[i] = self.Dx.dot(vec[0, i])
                out[i] += self.Dy.dot(vec[1, i].T).T

            return out

    def grad_utility(self, phi, param, work):
        """Compute ∇U for each species."""

        kappa = param["kappa"]
        Gamma = param["Gamma"]

        pi = work["pi"]
        dUdx = work["dUdx"]
        grad_pi = work["grad_pi"]
        grad_lap_phi = work["grad_lap_phi"]

        # pi[a] = sum_b kappa[a,b] phi[b]
        np.einsum("ab,bij->aij", kappa, phi, out=pi)

        self.grad(pi, out=grad_pi)
        self.grad_lapl(phi, out=grad_lap_phi)

        # dUdx[x,a] = sum_b Gamma[a,b] grad_lap_phi[x,b]
        np.einsum("ab,xbij->xaij", Gamma, grad_lap_phi, out=dUdx)

        dUdx += grad_pi

        nu = param.get("nu", None)
        if nu is not None and np.any(nu):
            # Keep this as a slow path for now.
            phisqr_ab = np.einsum("aij,bij->abij", phi, phi)
            nuterm = np.einsum("abc,bcij->aij", nu, phisqr_ab)
            dUdx += self.grad(nuterm)

        return dUdx
    
    def grad_utility_from_lap(self, phi, lap_phi, param, work):
        """Compute ∇U for 2 species,avoids calling grad_lapl"""
        pi = work["pi"]
        dUdx = work["dUdx"]
        grad_pi = work["grad_pi"]
        grad_lap_phi = work["grad_lap_phi"]

        kappa = param["kappa"]
        Gamma = param["Gamma"]

        pi[0] = kappa[0, 0] * phi[0] + kappa[0, 1] * phi[1]
        pi[1] = kappa[1, 0] * phi[0] + kappa[1, 1] * phi[1]

        self.grad(pi, out=grad_pi)
        self.grad(lap_phi, out=grad_lap_phi)

        dUdx[0, 0] = Gamma[0, 0] * grad_lap_phi[0, 0] + Gamma[0, 1] * grad_lap_phi[0, 1]
        dUdx[1, 0] = Gamma[0, 0] * grad_lap_phi[1, 0] + Gamma[0, 1] * grad_lap_phi[1, 1]

        dUdx[0, 1] = Gamma[1, 0] * grad_lap_phi[0, 0] + Gamma[1, 1] * grad_lap_phi[0, 1]
        dUdx[1, 1] = Gamma[1, 0] * grad_lap_phi[1, 0] + Gamma[1, 1] * grad_lap_phi[1, 1]

        dUdx += grad_pi
        return dUdx
    
    def utility_from_lap(self, phi, lap_phi, param, work):
        """
        Compute cell-centered utility-like field:

            U_a = sum_b kappa_ab phi_b + sum_b Gamma_ab lap_phi_b

        U shape: (nspecies, Nx, Ny)
        """
        kappa = param["kappa"]
        Gamma = param["Gamma"]

        U = work["U"]

        # Explicit 2-species version
        U[0] = kappa[0, 0] * phi[0] + kappa[0, 1] * phi[1]
        U[1] = kappa[1, 0] * phi[0] + kappa[1, 1] * phi[1]

        U[0] += Gamma[0, 0] * lap_phi[0] + Gamma[0, 1] * lap_phi[1]
        U[1] += Gamma[1, 0] * lap_phi[0] + Gamma[1, 1] * lap_phi[1]

        nu = param.get("nu", None)
        if nu is not None and np.any(nu):
            # Optional slow path if you still need nu.
            phisqr_ab = np.einsum("aij,bij->abij", phi, phi)
            nuterm = np.einsum("abc,bcij->aij", nu, phisqr_ab)
            U += nuterm

        return U

    def schelling_fv_flux_periodic(self, phi, phi0, U, param, work):
        """
        Compute face fluxes for div( M grad U ), periodic only.

        M_a = phi_a * phi0.

        det_flux_x[a, i, j] is the x-face flux through face i+1/2.
        det_flux_y[a, i, j] is the y-face flux through face j+1/2.
        """
        M = work["mobility"]
        Fx = work["det_flux_x"]
        Fy = work["det_flux_y"]

        # M[a] = phi[a] * phi0
        np.multiply(phi, phi0[np.newaxis, :, :], out=M)

        # Species 0 x-flux
        Fx[0, :-1, :] = 0.5 * (M[0, :-1, :] + M[0, 1:, :])
        Fx[0, :-1, :] *= (U[0, 1:, :] - U[0, :-1, :]) / self.dx

        Fx[0, -1, :] = 0.5 * (M[0, -1, :] + M[0, 0, :])
        Fx[0, -1, :] *= (U[0, 0, :] - U[0, -1, :]) / self.dx

        # Species 1 x-flux
        Fx[1, :-1, :] = 0.5 * (M[1, :-1, :] + M[1, 1:, :])
        Fx[1, :-1, :] *= (U[1, 1:, :] - U[1, :-1, :]) / self.dx

        Fx[1, -1, :] = 0.5 * (M[1, -1, :] + M[1, 0, :])
        Fx[1, -1, :] *= (U[1, 0, :] - U[1, -1, :]) / self.dx

        # Species 0 y-flux
        Fy[0, :, :-1] = 0.5 * (M[0, :, :-1] + M[0, :, 1:])
        Fy[0, :, :-1] *= (U[0, :, 1:] - U[0, :, :-1]) / self.dy

        Fy[0, :, -1] = 0.5 * (M[0, :, -1] + M[0, :, 0])
        Fy[0, :, -1] *= (U[0, :, 0] - U[0, :, -1]) / self.dy

        # Species 1 y-flux
        Fy[1, :, :-1] = 0.5 * (M[1, :, :-1] + M[1, :, 1:])
        Fy[1, :, :-1] *= (U[1, :, 1:] - U[1, :, :-1]) / self.dy

        Fy[1, :, -1] = 0.5 * (M[1, :, -1] + M[1, :, 0])
        Fy[1, :, -1] *= (U[1, :, 0] - U[1, :, -1]) / self.dy

        return Fx, Fy

    def schelling_fv_flux_neumann(self, phi, phi0, U, param, work):
        """
        Compute face fluxes for div( M grad U ), Neumann/no-normal-flux.
        """
        M = work["mobility"]
        Fx = work["det_flux_x"]
        Fy = work["det_flux_y"]

        np.multiply(phi, phi0[np.newaxis, :, :], out=M)

        Fx[:, 0, :] = 0.0
        Fx[:, -1, :] = 0.0
        Fy[:, :, 0] = 0.0
        Fy[:, :, -1] = 0.0

        # x interior faces, face index i = 1..Nx-1 between cells i-1 and i
        Fx[:, 1:-1, :] = 0.5 * (M[:, :-1, :] + M[:, 1:, :])
        Fx[:, 1:-1, :] *= (U[:, 1:, :] - U[:, :-1, :]) / self.dx

        # y interior faces, face index j = 1..Ny-1 between cells j-1 and j
        Fy[:, :, 1:-1] = 0.5 * (M[:, :, :-1] + M[:, :, 1:])
        Fy[:, :, 1:-1] *= (U[:, :, 1:] - U[:, :, :-1]) / self.dy

        return Fx, Fy
    
    def schelling_fv_div_M_grad_U(self, phi, phi0, U, param, work, out):
        """
        Compute out = div( M grad U ) using finite-volume face fluxes.
        """
        if self.bc == "periodic":
            self.schelling_fv_flux_periodic(phi, phi0, U, param, work)
        elif self.bc == "Neumann":
            self.schelling_fv_flux_neumann(phi, phi0, U, param, work)
        else:
            raise ValueError(
                f"Finite-volume Schelling flux not implemented for bc={self.bc}"
            )

        self.div_face_flux(
            work["det_flux_x"],
            work["det_flux_y"],
            out=out,
            work=work,
        )

        return out
        
    def conservative_face_noise_flux(self, phi, phi0, param, dt, work):
        """
        Fill work["noise_flux_x"] and work["noise_flux_y"] with conservative
        stochastic fluxes on cell faces.

        Periodic:
            noise_flux_x shape = (nspecies, Nx, Ny)
            noise_flux_y shape = (nspecies, Nx, Ny)

        Neumann:
            noise_flux_x shape = (nspecies, Nx+1, Ny)
            noise_flux_y shape = (nspecies, Nx, Ny+1)
            boundary normal fluxes are set to zero.
        """
        D = param["D"]
        h = param.get("h", np.sqrt(self.dx * self.dy))

        rho_center = work["rho_center"]
        amp_x = work["noise_amp_x"]
        amp_y = work["noise_amp_y"]
        flux_x = work["noise_flux_x"]
        flux_y = work["noise_flux_y"]

        inv_sqrt_cell = 1.0 / np.sqrt(self.dx * self.dy)

        # rho_center[a] = phi[a] * phi0
        np.multiply(phi, phi0[np.newaxis, :, :], out=rho_center)

        if self.bc == "periodic":
            # x-faces: average center mobility between i and i+1
            amp_x[:, :-1, :] = 0.5 * (rho_center[:, :-1, :] + rho_center[:, 1:, :])
            amp_x[:, -1, :] = 0.5 * (rho_center[:, -1, :] + rho_center[:, 0, :])

            # y-faces: average center mobility between j and j+1
            amp_y[:, :, :-1] = 0.5 * (rho_center[:, :, :-1] + rho_center[:, :, 1:])
            amp_y[:, :, -1] = 0.5 * (rho_center[:, :, -1] + rho_center[:, :, 0])

        elif self.bc == "Neumann":
            # Interior x-faces
            amp_x[:, 1:-1, :] = 0.5 * (rho_center[:, :-1, :] + rho_center[:, 1:, :])

            # Boundary x-faces: no normal flux
            amp_x[:, 0, :] = 0.0
            amp_x[:, -1, :] = 0.0

            # Interior y-faces
            amp_y[:, :, 1:-1] = 0.5 * (rho_center[:, :, :-1] + rho_center[:, :, 1:])

            # Boundary y-faces: no normal flux
            amp_y[:, :, 0] = 0.0
            amp_y[:, :, -1] = 0.0

        else:
            raise ValueError(
                f"Conservative face noise not implemented for bc={self.bc}"
            )

        # Apply floor before sqrt
        np.maximum(amp_x, 0.0, out=amp_x)
        np.maximum(amp_y, 0.0, out=amp_y)

        # Draw standard normals directly into the flux arrays
        self.rng.standard_normal(out=flux_x)
        self.rng.standard_normal(out=flux_y)

        # Convert amp arrays from mobility to full noise amplitude.
        # amp <- h * sqrt(2 D_a amp / dt) / sqrt(dx dy)
        for a in range(self.nspecies):
            amp_x[a] *= 2.0 * D[a] / dt
            amp_y[a] *= 2.0 * D[a] / dt

            np.sqrt(amp_x[a], out=amp_x[a])
            np.sqrt(amp_y[a], out=amp_y[a])

            amp_x[a] *= h * inv_sqrt_cell
            amp_y[a] *= h * inv_sqrt_cell

            flux_x[a] *= amp_x[a]
            flux_y[a] *= amp_y[a]

        if self.bc == "Neumann":
            # Make this explicit even though amplitudes are already zero there.
            flux_x[:, 0, :] = 0.0
            flux_x[:, -1, :] = 0.0
            flux_y[:, :, 0] = 0.0
            flux_y[:, :, -1] = 0.0

    def div_face_flux(self, flux_x, flux_y, out, work):
        """
        Compute finite-volume divergence of face fluxes.

        Periodic:
            flux_x[a, i, j] is the x-flux through the right face of cell (i,j)
            flux_y[a, i, j] is the y-flux through the top face of cell (i,j)

            out[a,i,j] =
                (Fx[a,i,j] - Fx[a,i-1,j]) / dx
            + (Fy[a,i,j] - Fy[a,i,j-1]) / dy

        Neumann:
            flux_x shape = (nspecies, Nx+1, Ny)
            flux_y shape = (nspecies, Nx, Ny+1)

            out[a,i,j] =
                (Fx[a,i+1,j] - Fx[a,i,j]) / dx
            + (Fy[a,i,j+1] - Fy[a,i,j]) / dy
        """
        if self.bc == "periodic":
            tmp = work["tmp"]

            # x contribution
            out[:, 1:, :] = flux_x[:, 1:, :] - flux_x[:, :-1, :]
            out[:, 0, :] = flux_x[:, 0, :] - flux_x[:, -1, :]
            out /= self.dx

            # y contribution into tmp
            tmp[:, :, 1:] = flux_y[:, :, 1:] - flux_y[:, :, :-1]
            tmp[:, :, 0] = flux_y[:, :, 0] - flux_y[:, :, -1]
            tmp /= self.dy

            out += tmp

        elif self.bc == "Neumann":
            out[:, :, :] = (
                (flux_x[:, 1:, :] - flux_x[:, :-1, :]) / self.dx
                + (flux_y[:, :, 1:] - flux_y[:, :, :-1]) / self.dy
            )

        else:
            raise ValueError(
                f"Face-flux divergence not implemented for bc={self.bc}"
            )

        return out

    def rhs_Vitelli(self, phi, param, dt, toggle_noise):
        """Compute RHS of the equation"""

        phi0   = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,)+self.N)
        dUdx = self.grad_utility(phi, param)
        
    
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        rhorho0dUdx = phi*phi0*dUdx
        div_dUdx = self.div2d(rhorho0dUdx) 
        D = param['D']
        lap_phi = self.lapl(phi)
        lap_phi0 = - lap_phi.sum(axis=0)
        divJ = np.einsum("a, aij->aij", D, phi0*lap_phi - phi*lap_phi0) - div_dUdx 
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            if 'h' in param:
                h = param['h']
            else:
                h = np.sqrt(self.dx*self.dy)

            xi = np.random.normal(0, 1, size= (2,)+phi.shape)/np.sqrt(self.dx*self.dy)
            if self.bc == 'Neumann':
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,:,0,:] = 0
                xi[:,:,-1,:] = 0
                xi[:,:,:,0] = 0
                xi[:,:,:,-1] = 0
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_term   = np.einsum("a, aij-> aij", param['D'], rho_face)*2/dt
            noise_flux   = np.einsum("aij, laij -> laij", h*np.sqrt(noise_term), xi) 
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += dnoise_dx
        
        return divJ

    def rhs_SchellingwithVoter(self, phi, param, dt, toggle_noise, work=None):
        """Compute RHS of the equation"""
        if work is None:
            work = self._ensure_work(phi.dtype)

        phi0 = work["phi0"]
        lap_phi = work["lap_phi"]
        lap_phi0 = work["lap_phi0"]
        flux = work["flux"]
        div_dUdx = work["div_dUdx"]
        divJ = work["divJ"]

        D = param["D"]
        beta = param["beta"]
        D_v = param["D_v"]

        self.lapl(phi, out=lap_phi)
        
        # phi0 = 1 - phi_A - phi_B
        np.sum(phi, axis=0, out=phi0)
        np.subtract(1.0, phi0, out=phi0)
        
        # lap_phi0 = -lap_phi.sum(axis=0)
        np.sum(lap_phi, axis=0, out=lap_phi0)
        np.negative(lap_phi0, out=lap_phi0)

        if self.schelling_flux == "collocated":
            dUdx = work["dUdx"]
            flux = work["flux"]

            self.grad_utility_from_lap(phi, lap_phi, param, work)
            # self.grad_utility(phi, param, work)

            # flux = phi * phi0 * dUdx
            np.multiply(dUdx, phi[np.newaxis, :, :, :], out=flux)
            flux *= phi0[np.newaxis, np.newaxis, :, :]

            self.div2d(flux, out=div_dUdx)

        elif self.schelling_flux == "finite_volume":
            U = self.utility_from_lap(phi, lap_phi, param, work)

            self.schelling_fv_div_M_grad_U(
                phi,
                phi0,
                U,
                param,
                work,
                out=div_dUdx,
            )

        else:
            raise ValueError(
                f"Unknown schelling_flux option: {self.schelling_flux}"
            )

        # divJ[0]
        divJ[0] = phi0 * lap_phi[0]
        divJ[0] -= phi[0] * lap_phi0
        divJ[0] -= beta * div_dUdx[0]
        divJ[0] *= D[0]

        # divJ[1]
        divJ[1] = phi0 * lap_phi[1]
        divJ[1] -= phi[1] * lap_phi0
        divJ[1] -= beta * div_dUdx[1]
        divJ[1] *= D[1]

        # voter current = D_v * (phi_B lap_phi_A - phi_A lap_phi_B)
        voter_current = work["voter_current"]
        np.multiply(phi[1], lap_phi[0], out=voter_current)
        voter_current -= phi[0] * lap_phi[1]
        voter_current *= D_v

        divJ[0] += voter_current
        divJ[1] -= voter_current
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            # Conservative Schelling/mobility noise as face fluxes
            self.conservative_face_noise_flux(phi, phi0, param, dt, work)

            self.div_face_flux(
                work["noise_flux_x"],
                work["noise_flux_y"],
                work["dnoise_dx"],
                work,
            )

            divJ += work["dnoise_dx"]

            # Demographic voter noise: keep as local non-conservative noise
            xi2 = work["xi2"]
            rho_ab = work["rho_ab"]
            demo_noise = work["demo_noise"]

            self.rng.standard_normal(out=xi2)
            xi2 *= 1.0 / np.sqrt(self.dx * self.dy)

            # rho_ab = max(phi_A * phi_B, floor^2)
            np.multiply(phi[0], phi[1], out=rho_ab)
            np.maximum(rho_ab, 0.0, out=rho_ab)

            # demo_noise = sqrt(4 d D_v rho_ab / dt) * xi2
            np.multiply(rho_ab, 4.0 * 2 * D_v / dt, out=demo_noise)
            np.sqrt(demo_noise, out=demo_noise)
            demo_noise *= xi2

            divJ[0] += demo_noise
            divJ[1] -= demo_noise
        
        return divJ
    
    def w0(self, phi, param):
        D = param["D"]
        beta = param["beta"]
        theta = param["theta"]
        sigma = param["sigma"]
        kappa = np.array([[theta-1,theta],[theta,theta-1]])
        Gamma = kappa*sigma**2/2
        pi = np.einsum("ab, bij-> aij", kappa, phi) + np.einsum("ab, bij-> aij", Gamma, self.lapl(phi))
        w0 = np.einsum("a, aij -> aij", D, 1/(1+np.exp(-beta*pi)))
        grad_pi = np.einsum("ab, lbij-> laij", kappa, self.grad(phi))
        grad_pi += np.einsum("ab, lbij-> laij", Gamma, self.grad_lapl(phi))
        gradw0 = np.einsum("aij, laij -> laij",  np.einsum("a, aij->aij", D, beta/(2+2*np.cosh(beta*pi))) , grad_pi)
        laplw0 = self.div2d(gradw0)
        return w0, gradw0, laplw0

    def rhs_Schelling(self, phi, param, dt, toggle_noise):
        """Compute RHS of the equation"""
        phi0   = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,)+self.N)
        w0, gradw0, laplw0 = self.w0(phi, param)
        
        """Compute grad J =   w0 (rho_0 ∂^2 rho -  rho ∂^2 rho_0)  + rho * rho_0 * ∂^2 w0 + 2 rho0 ∂ w0 . ∂ rho ) """
        gradphi = self.grad(phi)
        lap_phi = self.lapl(phi)
        lap_phi0 = - lap_phi.sum(axis=0)
        divJ = w0*phi0*lap_phi - w0*phi*lap_phi0  + phi*phi0*laplw0 + 2*phi0*(gradw0[0]*gradphi[0] + gradw0[1]*gradphi[1])
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            if 'h' in param:
                h = param['h']
            else:
                h = np.sqrt(self.dx*self.dy)
            xi = np.random.normal(0, 1, size= (2,) + phi.shape)/np.sqrt(self.dx*self.dy)
            if self.bc == 'Neumann':
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,:,0,:] = 0
                xi[:,:,-1,:] = 0
                xi[:,:,:,0] = 0
                xi[:,:,:,-1] = 0
            rho_face     = np.maximum(w0*phi*phi0, 1e-14) 
            noise_flux   = np.einsum("aij, laij -> laij", h*np.sqrt(2*rho_face/dt), xi)
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += dnoise_dx
        
        return divJ
        
    def step(self, rhs, phi, param, dt, toggle_noise, scheme, work = None, 
             step=None,
             record_projection_history=False,):
        phi_tot = np.sum(phi, axis=0)
        dphidt = rhs(phi, param, dt, toggle_noise, work)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred # + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            dphidt = dphidt.copy()
            rho_corr = phi + 0.5*dt*(dphidt + rhs(rho_pred, param,  dt, toggle_noise, work))
            rho_next = rho_corr # + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "RK4":
            dphidt = dphidt.copy()
            k1 = dt * dphidt
            k2 = dt * rhs(phi + 1/2* k1, param, dt, toggle_noise, work).copy()
            k3 = dt * rhs(phi + 1/2* k2, param, dt, toggle_noise, work).copy()
            k4 = dt * rhs(phi + k3, param, dt, toggle_noise, work)
            rho_next = phi + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            # rho_next +=  dt * phi*(param['b']*(1-phi_tot) - param['d'])

        return self.project_density(
                        rho_next,
                        work=work,
                        step=step,
                        record_history=record_projection_history,
                    )

    def scale_down(self, phi):
        ''' Scales down phi such that  sum_a phi[a] < 1 for all x 
        '''
        sumphi = np.sum(phi, axis=0)
        if np.any(sumphi >= 1 - self.phi_floor):
            # scale down slightly
            phi *= (1.0 / (sumphi.max() + self.phi_floor))
        return phi

    def _warmup_projection_kernels(self, phi, work=None):
        if not getattr(self, "use_numba_projection", False):
            return

        if work is None:
            work = self._ensure_work(phi.dtype)

        rho_tmp = phi.copy()

        diag_tmp = self._init_projection_diagnostics()

        # Force tiny artificial violations so both kernels compile.
        rho_tmp[0, 0, 0] = -1e-12
        rho_tmp[0, 0, 1] += 1e-12

        rho_tmp[0, 1, 1] = 0.7
        rho_tmp[1, 1, 1] = 0.7

        self._project_density_redistribute(
            rho_tmp,
            diag_tmp,
            projection_floor=getattr(self, "projection_floor", 0.0),
        )

    def _use_neumann_fv_fe_fastpath(self, method):
        return (
            method == "FE"
            and self.bc == "Neumann"
            and self.schelling_flux == "finite_volume"
            and not self.fft
        )

    def _run_neumann_fv_fe_fast(
        self,
        phi_init,
        param,
        nsteps,
        dt,
        noise=True,
        save_every=None,
        record_projection_history=False,
    ):
        """
        Fast production path for:
            - Neumann boundaries
            - finite-volume Schelling flux
            - forward Euler
            - optional stochastic noise
            - adaptive projection
        """
        work = self._ensure_work(phi_init.dtype)

        phi = phi_init.copy()
        phi_next = work.get("phi_next", None)

        if phi_next is None or phi_next.shape != phi.shape:
            work["phi_next"] = np.empty_like(phi)
            phi_next = work["phi_next"]

        # Optional output storage.
        phi_run = []
        if save_every is not None:
            phi_run.append(phi.copy())

        # Warm up Numba projection kernels before production timing if desired.
        if getattr(self, "use_numba_projection", False):
            self._warmup_projection_kernels(phi, work)

        for step in range(1, nsteps + 1):
            rhs = self.rhs_SchellingwithVoter(
                phi,
                param,
                dt=dt,
                toggle_noise=noise,
                work=work,
            )

            # Forward Euler update.
            np.copyto(phi_next, phi)
            phi_next += dt * rhs

            save_this_step = (
                save_every is not None
                and step % save_every == 0
            )

            self.project_density(
                phi_next,
                work=work,
                step=step,
                record_history=(
                    record_projection_history
                    and save_this_step
                ),
            )

            # Swap buffers.
            phi, phi_next = phi_next, phi

            if save_this_step:
                phi_run.append(phi.copy())

        if save_every is None:
            return phi

        return np.asarray(phi_run).transpose((1,0,2,3))

    def run(self, phi, param, nsteps, dt, toggle_noise, 
            no_frames = 100,
            scheme = "FE", 
            model = "Schelling+Voter",
            verbatum = True,
            diagnostic_interval=100,
            reset_projection_diag=True,
            use_fastpath=False):
        ''' Runs the FHD simulation with specified parameters for nsteps, recording no_frames equally timed frames.

        Arg:
            phi:    Initial condition for phi, should have shape (nspecies, N)
            param:  Dictionary with parameter settings, see 'model' below for expected parameters
            nsteps: Number of simulation steps
            dt:     Size of the time step
            toggle_noise: strength of noise term, if zero, no noise is used
            no_frames: Number of frames saved in the final np. array
            scheme: Numerical integration scheme, choose between forward Euler 'FE' or predictor-corrector 'PC'
            model:  Specify which model to run. Options are: 
                    "Vitelli"  with expected parameters in dictionary param:
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters \kappa^{ab} \phi_b
                        'Gamma': numpy array of shape (nspecies, nspecies) for the lapl(phi) term in the utility \Gamma^{ab} \nabla^2 \phi_b
                        'nu':    Optional np.array of shape (nspecies, nspecies, nspecies) for the quadratic term in the utility: nu^{abc} \phi_b \phi_c
                    "Schelling" with expected parameters in dictionary param:
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'theta': float: satisfaction threshold (between 0 and 1)
                        'sigma': Coefficient of the lapl(phi) term in pi (sigma^2/2 of the Gaussian neighborhood kernel)
                        'beta':  Inverse temperature, large beta means scricter enforcement of threshold moves
                    "Schelling+Voter" with expected parameters
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters \kappa^{ab} \phi_b
                        'Gamma': numpy array of shape (nspecies, nspecies) for the lapl(phi) term in the utility \Gamma^{ab} \nabla^2 \phi_b
                        'nu':    Optional np.array of shape (nspecies, nspecies, nspecies) for the quadratic term in the utility: nu^{abc} \phi_b \phi_c
                        'D_v':   float: coefficient for voter model diffusion term
                        'noise_v': float: strength of voter model noise (default is one)
            verbatum: If True print and plot stuff

        Returns:
            phi_run: numpy array of shape (nspecies, frames+1, N) with the simulation timeseries
                    
        '''
        plot_every = nsteps//no_frames
        phi = self.project_density_fast(phi)
        phi = np.maximum(phi, self.phi_floor)

        work = self._ensure_work(phi.dtype)
        
        if reset_projection_diag:
            work["projection_diag"] = self._init_projection_diagnostics()

        phi_current = phi.copy()

        if use_fastpath and self._use_neumann_fv_fe_fastpath(scheme):
            return self._run_neumann_fv_fe_fast(
                phi_init=phi_current,
                param=param,
                nsteps=nsteps,
                dt=dt,
                noise=toggle_noise,
                save_every=plot_every,
                record_projection_history=(diagnostic_interval is not None),
            )
        
        phi_run = np.zeros((self.nspecies, no_frames+1)+self.N)
        phi_run[:,0,:,:] = phi_current
        
        for n in range(1, nsteps + 1):     
            # print("step", n)
            if model == "Vitelli":
                phi_current = self.step(self.rhs_Vitelli, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling":
                phi_current = self.step(self.rhs_Schelling, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling+Voter":
                record_projection_history = (
                    diagnostic_interval is not None
                    and n % diagnostic_interval == 0
                )

                phi_current = self.step(
                    self.rhs_SchellingwithVoter,
                    phi_current,
                    param,
                    dt,
                    toggle_noise,
                    scheme,
                    work=work,
                    step=n,
                    record_projection_history=record_projection_history,
                )
                # phi_current = self.step(self.rhs_SchellingwithVoter, phi_current, param, dt, toggle_noise, scheme, work)
            else:
                raise ValueError(f"Model {model} is unknown, please choose 'Vitelli', 'Schelling' or 'Schelling+Voter'") 
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {dissimilarity(phi_current):.6f}, mean_kl_div = {mean_relative_entropy(phi_current):.6f}")
                phi_run[:,n//plot_every,:,:] = phi_current

        if verbatum:
            phi_diff = phi_run[0,:,:,:] - phi_run[1,:,:,:]
            im =plt.imshow(phi_diff[-1], cmap = 'RdBu', aspect='auto', origin='lower', extent=[-self.Lx/2,self.Lx/2,-self.Ly/2,self.Ly/2], vmin=-1, vmax=1)
            kappa = param['kappa']
            D = param['D']
            Gamma = param['Gamma']
            D_v = param['D_v']
            title = fr"$D = [{D[0]:.1f}, {D[1]:.1f}],\, \kappa = [[{kappa[0,0]:.1f}, {kappa[0,1]:.1f}], [{kappa[1,0]:.1f} , {kappa[1,1]:.1f}]] \,, \Gamma = [[{Gamma[0,0]:.1f}, {Gamma[0,1]:.1f}], [{Gamma[1,0]:.1f} , {Gamma[1,1]:.1f}]], \, D_v = {D_v} $"
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            cbar = plt.colorbar(im, fraction=0.046)
            cbar.set_label(r"$\phi_a - \phi_b$",size=14)
            plt.show()
            
        return phi_run
    
    def run_until_converged(self, phi, param, dt, toggle_noise, 
            save_every = 500, T= 100, K = 50,
            scheme = "FE", 
            model = "Vitelli",
            verbatum = True):
        ''' Runs the FHD simulation with specified parameters until converged, recording every save_every timesteps.

        Arg:
            phi:    Initial condition for phi, should have shape (nspecies, N)
            param:  Dictionary with parameter settings, see 'model' below for expected parameters
            dt:     Size of the time step
            toggle_noise: strength of noise term, if zero, no noise is used
            save_every: Number of frames between storing a snapshot in the final np.array
            T:      int: number of snapshots taking into the time-window for computing the slidding averages
            K:      int: number of snapshots over which the change in slidding window averages is below convergence threshold 
            scheme: Numerical integration scheme, choose between forward Euler 'FE' or predictor-corrector 'PC'
            model:  Specify which model to run. Options are: 
                    "Vitelli"  with expected parameters in dictionary param:
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters \kappa^{ab} \phi_b
                        'Gamma': numpy array of shape (nspecies, nspecies) for the lapl(phi) term in the utility \Gamma^{ab} \nabla^2 \phi_b
                        'nu':    Optional np.array of shape (nspecies, nspecies, nspecies) for the quadratic term in the utility: nu^{abc} \phi_b \phi_c
                    "Schelling" with expected parameters in dictionary param:
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'theta': float: satisfaction threshold (between 0 and 1)
                        'sigma': Coefficient of the lapl(phi) term in pi (sigma^2/2 of the Gaussian neighborhood kernel)
                        'beta':  Inverse temperature, large beta means scricter enforcement of threshold moves
                    "Schelling+Voter" with expected parameters
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters \kappa^{ab} \phi_b
                        'Gamma': numpy array of shape (nspecies, nspecies) for the lapl(phi) term in the utility \Gamma^{ab} \nabla^2 \phi_b
                        'nu':    Optional np.array of shape (nspecies, nspecies, nspecies) for the quadratic term in the utility: nu^{abc} \phi_b \phi_c
                        'D_v':   float: coefficient for voter model diffusion term
            verbatum: If True print and plot stuff

        Returns:
            phi_run: numpy array of shape (nspecies, frames+1, N) with the simulation timeseries
                    
        '''
        plot_every = save_every
        phi = self.project_density_fast(phi)
        phi = np.maximum(phi, self.phi_floor)
        
        phi_current = phi.copy()
        phi_run = [phi_current]
        
        converged = False
        eps_mean = 1e-3

        DKL = []
        Dis_idx = []
        H_idx = []
        n = 0

        while not converged:
            # print("step", n)
            n+=1
            if model == "Vitelli":
                phi_current = self.step(self.rhs_Vitelli, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling":
                phi_current = self.step(self.rhs_Schelling, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling+Voter":
                phi_current = self.step(self.rhs_SchellingwithVoter, phi_current, param, dt, toggle_noise, scheme)
            else:
                raise ValueError(f"Model {model} is unknown, please choose 'Vitelli', 'Schelling' or 'Schelling+Voter'") 
        
            if n % plot_every == 0:
                Dis_idx.append(dissimilarity(phi_current))
                DKL.append(mean_relative_entropy(phi_current))
                if verbatum:
                    print(f"Step {n}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {Dis_idx[-1]:.6f}, KL_divergence = {DKL[-1]:.6f}")
                phi_run.append(phi_current)

                if len(DKL)>T+K:
                    converged = check_convergence([Dis_idx, DKL], T, eps_mean, K)
                    if Dis_idx[-1] < eps_mean: # If dissimilarity is very small, run converged in well-mixed state
                        converged = True

        phi_run = np.array(phi_run).transpose(1,0,2,3)
        
        if verbatum:
            phi_diff = phi_run[0,:,:,:] - phi_run[1,:,:,:]
            im =plt.imshow(phi_diff[-1], cmap = 'RdBu', aspect='auto', origin='lower', extent=[-self.Lx/2,self.Lx/2,-self.Ly/2,self.Ly/2], vmin=-1, vmax=1)
            kappa = param['kappa']
            D = param['D']
            Gamma = param['Gamma']
            D_v = param['D_v']
            title = fr"$D = [{D[0]:.1f}, {D[1]:.1f}],\, \kappa = [[{kappa[0,0]:.1f}, {kappa[0,1]:.1f}], [{kappa[1,0]:.1f} , {kappa[1,1]:.1f}]] \,, \Gamma = [[{Gamma[0,0]:.1f}, {Gamma[0,1]:.1f}], [{Gamma[1,0]:.1f} , {Gamma[1,1]:.1f}]], \, D_v = {D_v} $"
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("t")
            cbar = plt.colorbar(im, fraction=0.046)
            cbar.set_label(r"$\phi_a - \phi_b$",size=14)
            plt.show()
        
        return phi_run

    def print_projection_diagnostics(self, work=None):
        if work is None:
            work = self._work

        diag = work["projection_diag"]

        expected_net = (
            diag["mass_added_low"]
            - diag["mass_removed_high"]
            - diag["mass_removed_simplex"]
            + diag.get("mass_added_transfer_fallback", 0.0)
            + diag.get("mass_roundoff_cleanup", 0.0)
        )

        print("Projection diagnostics")
        print("----------------------")
        print(f"calls:                 {diag['n_calls']}")

        print(f"low entries:           {diag['n_low_entries']}")
        print(f"mass added low:        {diag['mass_added_low']:.8e}")
        print(f"max low violation:     {diag['max_low_violation']:.8e}")

        print(f"high entries:          {diag['n_high_entries']}")
        print(f"mass removed high:     {diag['mass_removed_high']:.8e}")
        print(f"max high violation:    {diag['max_high_violation']:.8e}")

        print(f"simplex cells:         {diag['n_simplex_cells']}")
        print(f"mass removed simplex:  {diag['mass_removed_simplex']:.8e}")
        print(f"max simplex violation: {diag['max_simplex_violation']:.8e}")

        print(f"net mass change:       {diag['net_mass_change_projection']:.8e}")
        print(f"expected net change:   {expected_net:.8e}")
        print(
            f"bookkeeping error:     "
            f"{diag['net_mass_change_projection'] - expected_net:.8e}"
        )

        print(f"mass transferred low:  {diag['mass_transferred_low']:.8e}")
        print(f"mass transferred AtoB: {diag['mass_transferred_AtoB']:.8e}")
        print(f"mass transferred BtoA: {diag['mass_transferred_BtoA']:.8e}")
        print(f"transfer fallback n:   {diag['n_transfer_fallback_entries']}")
        print(f"transfer fallback add: {diag['mass_added_transfer_fallback']:.8e}")

        print(f"redis low local:       {diag['mass_redistributed_low_local']:.8e}")
        print(f"redis low global:      {diag['mass_redistributed_low_global']:.8e}")
        print(f"redis high local:      {diag['mass_redistributed_high_local']:.8e}")
        print(f"redis high global:     {diag['mass_redistributed_high_global']:.8e}")
        print(f"redis simplex local:   {diag['mass_redistributed_simplex_local']:.8e}")
        print(f"redis simplex global:  {diag['mass_redistributed_simplex_global']:.8e}")

        print(f"redis low leftover:    {diag['redistribute_low_leftover']:.8e}")
        print(f"redis high leftover:   {diag['redistribute_high_leftover']:.8e}")
        print(f"redis simplex leftover:{diag['redistribute_simplex_leftover']:.8e}")

        print(f"redis low failures:    {diag['n_redistribute_low_failures']}")
        print(f"redis high failures:   {diag['n_redistribute_high_failures']}")
        print(f"redis simplex failures:{diag['n_redistribute_simplex_failures']}")

        print(f"no. expensive proj:    {diag['n_expensive_projection_calls']}")
        print(f"no. roundoff cleanups: {diag['n_roundoff_cleanup_calls']}")
        print(f"mass roundoff cleanup: {diag['mass_roundoff_cleanup']:.8e}")


    def projection_history_as_dict(self, work=None):
        if work is None:
            work = self._work

        hist = work["projection_diag"]["history"]

        if len(hist) == 0:
            return {}

        keys = hist[0].keys()
        return {
            key: np.array([h[key] for h in hist])
            for key in keys
        }