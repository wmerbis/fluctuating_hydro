#!/usr/bin/env python3
"""
Optimized conservative Cartesian finite-volume solver for the stochastic
Schelling+Voter hydrodynamics on a regular 2D grid.

Main optimizations compared to the baseline prototype
----------------------------------------------------
1. Vectorized semi-implicit stiff step in Fourier space.
   The previous Python loop over all Fourier modes has been replaced by an
   analytic batched inversion of the 2x2 Fourier-space systems.

2. Numba-jitted explicit deterministic RHS and stochastic increment kernels.
   The hot path now uses conservative face-based updates in compiled loops.

3. Reduced memory allocation.
   Large work arrays are allocated once in __init__ and then filled in-place.

Noise implementation
--------------------
* Schelling conservative noise: one Gaussian draw per face, shared by the two
  adjacent cells. This makes the discrete Schelling noise exactly conservative.
* Voter demographic noise: one Gaussian draw per cell, reused with opposite
  sign for species A and B. Hence occupied mass rho_A + rho_B is preserved
  exactly up to floating-point roundoff.

Notes
-----
* The explicit step supports an active mask and either periodic or no-flux BC.
* The semi-implicit Gamma step is currently implemented for periodic BC and a
  fully active regular grid only.
* This remains a research prototype: no positivity/simplex limiter is included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco

Array = np.ndarray


@njit(cache=True)
def _zero_array(a: Array) -> None:
    ny, nx = a.shape
    for iy in range(ny):
        for ix in range(nx):
            a[iy, ix] = 0.0


@njit(cache=True)
def _compute_rho0_inplace(rho_A: Array, rho_B: Array, active: Array, rho0: Array) -> None:
    ny, nx = rho_A.shape
    for iy in range(ny):
        for ix in range(nx):
            if active[iy, ix]:
                rho0[iy, ix] = 1.0 - rho_A[iy, ix] - rho_B[iy, ix]
            else:
                rho0[iy, ix] = 0.0


@njit(cache=True)
def _compute_linear_utility_inplace(
    rho_A: Array,
    rho_B: Array,
    active: Array,
    k00: float,
    k01: float,
    k10: float,
    k11: float,
    U_A: Array,
    U_B: Array,
) -> None:
    ny, nx = rho_A.shape
    for iy in range(ny):
        for ix in range(nx):
            if active[iy, ix]:
                ra = rho_A[iy, ix]
                rb = rho_B[iy, ix]
                U_A[iy, ix] = k00 * ra + k01 * rb
                U_B[iy, ix] = k10 * ra + k11 * rb
            else:
                U_A[iy, ix] = 0.0
                U_B[iy, ix] = 0.0


@njit(cache=True)
def _explicit_rhs_numba(
    rho_A: Array,
    rho_B: Array,
    active: Array,
    dx: float,
    dy: float,
    bc_periodic: int,
    D_A: float,
    D_B: float,
    D_v: float,
    beta: float,
    k00: float,
    k01: float,
    k10: float,
    k11: float,
    rho0: Array,
    U_A: Array,
    U_B: Array,
    rhs_A: Array,
    rhs_B: Array,
) -> None:
    ny, nx = rho_A.shape

    _compute_rho0_inplace(rho_A, rho_B, active, rho0)
    _compute_linear_utility_inplace(rho_A, rho_B, active, k00, k01, k10, k11, U_A, U_B)
    _zero_array(rhs_A)
    _zero_array(rhs_B)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    # x-faces: cell (iy, ix) with its right neighbor
    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                if bc_periodic == 1:
                    jx = 0
                else:
                    continue

            if not (active[iy, ix] and active[iy, jx]):
                continue

            rhoA_x = 0.5 * (rho_A[iy, ix] + rho_A[iy, jx])
            rhoB_x = 0.5 * (rho_B[iy, ix] + rho_B[iy, jx])
            rho0_x = 0.5 * (rho0[iy, ix] + rho0[iy, jx])

            d_rhoA_x = (rho_A[iy, jx] - rho_A[iy, ix]) * inv_dx
            d_rhoB_x = (rho_B[iy, jx] - rho_B[iy, ix]) * inv_dx
            d_rho0_x = (rho0[iy, jx] - rho0[iy, ix]) * inv_dx
            d_UA_x = (U_A[iy, jx] - U_A[iy, ix]) * inv_dx
            d_UB_x = (U_B[iy, jx] - U_B[iy, ix]) * inv_dx

            mobA_x = rho0_x * rhoA_x
            mobB_x = rho0_x * rhoB_x

            F_A = D_A * (rho0_x * d_rhoA_x - rhoA_x * d_rho0_x - beta * mobA_x * d_UA_x)
            F_A += D_v * (rhoB_x * d_rhoA_x - rhoA_x * d_rhoB_x)

            F_B = D_B * (rho0_x * d_rhoB_x - rhoB_x * d_rho0_x - beta * mobB_x * d_UB_x)
            F_B += D_v * (rhoA_x * d_rhoB_x - rhoB_x * d_rhoA_x)

            contrib_A = F_A * inv_dx
            contrib_B = F_B * inv_dx
            rhs_A[iy, ix] += contrib_A
            rhs_A[iy, jx] -= contrib_A
            rhs_B[iy, ix] += contrib_B
            rhs_B[iy, jx] -= contrib_B

    # y-faces: cell (iy, ix) with its upward neighbor
    for iy in range(ny):
        jy = iy + 1
        for ix in range(nx):
            ky = jy
            if ky == ny:
                if bc_periodic == 1:
                    ky = 0
                else:
                    continue

            if not (active[iy, ix] and active[ky, ix]):
                continue

            rhoA_y = 0.5 * (rho_A[iy, ix] + rho_A[ky, ix])
            rhoB_y = 0.5 * (rho_B[iy, ix] + rho_B[ky, ix])
            rho0_y = 0.5 * (rho0[iy, ix] + rho0[ky, ix])

            d_rhoA_y = (rho_A[ky, ix] - rho_A[iy, ix]) * inv_dy
            d_rhoB_y = (rho_B[ky, ix] - rho_B[iy, ix]) * inv_dy
            d_rho0_y = (rho0[ky, ix] - rho0[iy, ix]) * inv_dy
            d_UA_y = (U_A[ky, ix] - U_A[iy, ix]) * inv_dy
            d_UB_y = (U_B[ky, ix] - U_B[iy, ix]) * inv_dy

            mobA_y = rho0_y * rhoA_y
            mobB_y = rho0_y * rhoB_y

            F_A = D_A * (rho0_y * d_rhoA_y - rhoA_y * d_rho0_y - beta * mobA_y * d_UA_y)
            F_A += D_v * (rhoB_y * d_rhoA_y - rhoA_y * d_rhoB_y)

            F_B = D_B * (rho0_y * d_rhoB_y - rhoB_y * d_rho0_y - beta * mobB_y * d_UB_y)
            F_B += D_v * (rhoA_y * d_rhoB_y - rhoB_y * d_rhoA_y)

            contrib_A = F_A * inv_dy
            contrib_B = F_B * inv_dy
            rhs_A[iy, ix] += contrib_A
            rhs_A[ky, ix] -= contrib_A
            rhs_B[iy, ix] += contrib_B
            rhs_B[ky, ix] -= contrib_B


@njit(cache=True)
def _stochastic_increment_numba(
    rho_A: Array,
    rho_B: Array,
    active: Array,
    dx: float,
    dy: float,
    vol: float,
    bc_periodic: int,
    D_A: float,
    D_B: float,
    D_v: float,
    h_noise: float,
    spatial_dim: int,
    dt: float,
    rho0: Array,
    dW_A: Array,
    dW_B: Array,
    eta_A_x: Array,
    eta_B_x: Array,
    eta_A_y: Array,
    eta_B_y: Array,
    xi: Array,
) -> None:
    ny, nx = rho_A.shape

    _compute_rho0_inplace(rho_A, rho_B, active, rho0)
    _zero_array(dW_A)
    _zero_array(dW_B)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    scale_cv = np.sqrt(dt / vol)

    # Conservative Schelling noise on x-faces
    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                if bc_periodic == 1:
                    jx = 0
                else:
                    continue

            if not (active[iy, ix] and active[iy, jx]):
                continue

            rhoA_x = 0.5 * (rho_A[iy, ix] + rho_A[iy, jx])
            rhoB_x = 0.5 * (rho_B[iy, ix] + rho_B[iy, jx])
            rho0_x = 0.5 * (rho0[iy, ix] + rho0[iy, jx])

            amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_x * rhoA_x, 0.0))
            amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_x * rhoB_x, 0.0))

            dF_A = scale_cv * amp_A * eta_A_x[iy, ix]
            dF_B = scale_cv * amp_B * eta_B_x[iy, ix]

            contrib_A = dF_A * inv_dx
            contrib_B = dF_B * inv_dx
            dW_A[iy, ix] += contrib_A
            dW_A[iy, jx] -= contrib_A
            dW_B[iy, ix] += contrib_B
            dW_B[iy, jx] -= contrib_B

    # Conservative Schelling noise on y-faces
    for iy in range(ny):
        ky0 = iy + 1
        for ix in range(nx):
            ky = ky0
            if ky == ny:
                if bc_periodic == 1:
                    ky = 0
                else:
                    continue

            if not (active[iy, ix] and active[ky, ix]):
                continue

            rhoA_y = 0.5 * (rho_A[iy, ix] + rho_A[ky, ix])
            rhoB_y = 0.5 * (rho_B[iy, ix] + rho_B[ky, ix])
            rho0_y = 0.5 * (rho0[iy, ix] + rho0[ky, ix])

            amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_y * rhoA_y, 0.0))
            amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_y * rhoB_y, 0.0))

            dF_A = scale_cv * amp_A * eta_A_y[iy, ix]
            dF_B = scale_cv * amp_B * eta_B_y[iy, ix]

            contrib_A = dF_A * inv_dy
            contrib_B = dF_B * inv_dy
            dW_A[iy, ix] += contrib_A
            dW_A[ky, ix] -= contrib_A
            dW_B[iy, ix] += contrib_B
            dW_B[ky, ix] -= contrib_B

    # Voter demographic noise: opposite sign source pair
    voter_prefactor = 2.0 * spatial_dim * D_v * (h_noise ** (spatial_dim - 2))
    for iy in range(ny):
        for ix in range(nx):
            if active[iy, ix]:
                amp_v = np.sqrt(max(voter_prefactor * rho_A[iy, ix] * rho_B[iy, ix], 0.0))
                source = scale_cv * amp_v * xi[iy, ix]
                dW_A[iy, ix] += source
                dW_B[iy, ix] -= source
            else:
                dW_A[iy, ix] = 0.0
                dW_B[iy, ix] = 0.0


@dataclass
class ModelParameters:
    D_A: float = 0.1
    D_B: float = 0.1
    D_v: float = 0.01
    beta: float = 10.0
    h_noise: float = 1.0
    spatial_dim: int = 2
    kappa: Array = None
    gamma: Array = None

    def __post_init__(self) -> None:
        if self.kappa is None:
            self.kappa = np.array([[0.6, -0.2], [-0.2, 0.6]], dtype=float)
        else:
            self.kappa = np.asarray(self.kappa, dtype=float)
        if self.gamma is None:
            self.gamma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        else:
            self.gamma = np.asarray(self.gamma, dtype=float)
        if self.kappa.shape != (2, 2):
            raise ValueError("kappa must have shape (2,2)")
        if self.gamma.shape != (2, 2):
            raise ValueError("gamma must have shape (2,2)")
        if self.spatial_dim != 2:
            raise NotImplementedError("This implementation currently assumes d = 2.")


class SchellingVoterFVSolverOptimized:
    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        params: ModelParameters,
        active: Optional[Array] = None,
        bc: str = "periodic",
        semi_implicit_stiff: bool = True,
        stochastic: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.vol = self.dx * self.dy
        self.params = params
        self.bc = bc.lower()
        self.bc_periodic = 1 if self.bc == "periodic" else 0
        self.semi_implicit_stiff = semi_implicit_stiff
        self.stochastic = stochastic
        self.rng = np.random.default_rng() if rng is None else rng

        if self.bc not in {"periodic", "noflux"}:
            raise ValueError("bc must be 'periodic' or 'noflux'")

        if active is None:
            self.active = np.ones((self.ny, self.nx), dtype=np.bool_)
        else:
            active = np.asarray(active, dtype=np.bool_)
            if active.shape != (self.ny, self.nx):
                raise ValueError("active mask must have shape (ny, nx)")
            self.active = active.copy()

        self.rho_A = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.rho_B = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Scratch buffers reused every step.
        self._rho0 = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._U_A = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._U_B = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rhs_A = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rhs_B = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rho_A_star = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rho_B_star = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._dW_A = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._dW_B = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rho_A_new = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._rho_B_new = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Random buffers; filled each step.
        self._eta_A_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._eta_B_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._eta_A_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._eta_B_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._xi = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Precompute Fourier wave numbers for the stiff step.
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self._KX, self._KY = np.meshgrid(kx, ky)
        self._k2 = self._KX**2 + self._KY**2
        self._k4 = self._k2**2
        self._zero_mode = self._k4 == 0.0

    def set_state(self, rho_A: Array, rho_B: Array) -> None:
        rho_A = np.asarray(rho_A, dtype=np.float64)
        rho_B = np.asarray(rho_B, dtype=np.float64)
        if rho_A.shape != (self.ny, self.nx) or rho_B.shape != (self.ny, self.nx):
            raise ValueError("rho_A and rho_B must have shape (ny, nx)")
        self.rho_A[...] = np.where(self.active, rho_A, 0.0)
        self.rho_B[...] = np.where(self.active, rho_B, 0.0)

    def rho_0(self) -> Array:
        _compute_rho0_inplace(self.rho_A, self.rho_B, self.active, self._rho0)
        return self._rho0.copy()

    def total_masses(self) -> Dict[str, float]:
        _compute_rho0_inplace(self.rho_A, self.rho_B, self.active, self._rho0)
        occ = self.rho_A + self.rho_B
        return {
            "A": float(np.sum(self.rho_A[self.active]) * self.vol),
            "B": float(np.sum(self.rho_B[self.active]) * self.vol),
            "vacancy": float(np.sum(self._rho0[self.active]) * self.vol),
            "occupied": float(np.sum(occ[self.active]) * self.vol),
            "total": float(np.sum((occ + self._rho0)[self.active]) * self.vol),
        }

    def explicit_rhs(self, rho_A: Array, rho_B: Array) -> Tuple[Array, Array]:
        _explicit_rhs_numba(
            rho_A,
            rho_B,
            self.active,
            self.dx,
            self.dy,
            self.bc_periodic,
            self.params.D_A,
            self.params.D_B,
            self.params.D_v,
            self.params.beta,
            self.params.kappa[0, 0],
            self.params.kappa[0, 1],
            self.params.kappa[1, 0],
            self.params.kappa[1, 1],
            self._rho0,
            self._U_A,
            self._U_B,
            self._rhs_A,
            self._rhs_B,
        )
        return self._rhs_A, self._rhs_B

    def stochastic_increment(self, rho_A: Array, rho_B: Array, dt: float) -> Tuple[Array, Array]:
        # Fill random buffers. These assignments still create temporaries from NumPy's RNG,
        # but the large deterministic work arrays are reused in-place.
        self._eta_A_x[...] = self.rng.standard_normal((self.ny, self.nx))
        self._eta_B_x[...] = self.rng.standard_normal((self.ny, self.nx))
        self._eta_A_y[...] = self.rng.standard_normal((self.ny, self.nx))
        self._eta_B_y[...] = self.rng.standard_normal((self.ny, self.nx))
        self._xi[...] = self.rng.standard_normal((self.ny, self.nx))

        _stochastic_increment_numba(
            rho_A,
            rho_B,
            self.active,
            self.dx,
            self.dy,
            self.vol,
            self.bc_periodic,
            self.params.D_A,
            self.params.D_B,
            self.params.D_v,
            self.params.h_noise,
            self.params.spatial_dim,
            dt,
            self._rho0,
            self._dW_A,
            self._dW_B,
            self._eta_A_x,
            self._eta_B_x,
            self._eta_A_y,
            self._eta_B_y,
            self._xi,
        )
        return self._dW_A, self._dW_B

    def _assert_stiff_step_supported(self) -> None:
        if self.bc != "periodic":
            raise NotImplementedError("Semi-implicit stiff step is currently implemented for periodic BC only.")
        if not np.all(self.active):
            raise NotImplementedError(
                "Semi-implicit stiff step is currently implemented for fully active regular grids only."
            )

    def semi_implicit_gamma_step(self, rho_A_star: Array, rho_B_star: Array, dt: float) -> Tuple[Array, Array]:
        self._assert_stiff_step_supported()

        rho0_star = 1.0 - rho_A_star - rho_B_star
        Mbar_A = self.params.beta * self.params.D_A * float(np.mean(rho0_star * rho_A_star))
        Mbar_B = self.params.beta * self.params.D_B * float(np.mean(rho0_star * rho_B_star))

        g = self.params.gamma
        alpha = dt * self._k4

        m11 = 1.0 + alpha * (Mbar_A * g[0, 0])
        m12 = alpha * (Mbar_A * g[0, 1])
        m21 = alpha * (Mbar_B * g[1, 0])
        m22 = 1.0 + alpha * (Mbar_B * g[1, 1])
        det = m11 * m22 - m12 * m21

        rhoA_hat = np.fft.fft2(rho_A_star)
        rhoB_hat = np.fft.fft2(rho_B_star)

        rhoA_new_hat = (m22 * rhoA_hat - m12 * rhoB_hat) / det
        rhoB_new_hat = (-m21 * rhoA_hat + m11 * rhoB_hat) / det

        # Preserve exact zero mode => exact conservation of species masses in stiff step.
        rhoA_new_hat[self._zero_mode] = rhoA_hat[self._zero_mode]
        rhoB_new_hat[self._zero_mode] = rhoB_hat[self._zero_mode]

        self._rho_A_new[...] = np.real(np.fft.ifft2(rhoA_new_hat))
        self._rho_B_new[...] = np.real(np.fft.ifft2(rhoB_new_hat))
        return self._rho_A_new, self._rho_B_new

    def step(self, dt: float, add_noise: Optional[bool] = None) -> None:
        if add_noise is None:
            add_noise = self.stochastic

        rhs_A, rhs_B = self.explicit_rhs(self.rho_A, self.rho_B)
        self._rho_A_star[...] = self.rho_A + dt * rhs_A
        self._rho_B_star[...] = self.rho_B + dt * rhs_B

        if add_noise:
            dW_A, dW_B = self.stochastic_increment(self.rho_A, self.rho_B, dt)
            self._rho_A_star += dW_A
            self._rho_B_star += dW_B

        if self.semi_implicit_stiff:
            rho_A_new, rho_B_new = self.semi_implicit_gamma_step(self._rho_A_star, self._rho_B_star, dt)
        else:
            rho_A_new = self._rho_A_star
            rho_B_new = self._rho_B_star

        self.rho_A[...] = np.where(self.active, rho_A_new, 0.0)
        self.rho_B[...] = np.where(self.active, rho_B_new, 0.0)

    def run(self, dt: float, nsteps: int, snapshot_every: int = 0, add_noise: Optional[bool] = None) -> Dict[str, Array]:
        masses_A = []
        masses_B = []
        masses_occ = []
        masses_tot = []
        times = []
        snapshots = []

        for n in range(nsteps + 1):
            m = self.total_masses()
            masses_A.append(m["A"])
            masses_B.append(m["B"])
            masses_occ.append(m["occupied"])
            masses_tot.append(m["total"])
            times.append(n * dt)

            if snapshot_every > 0 and n % snapshot_every == 0:
                snapshots.append((n * dt, self.rho_A.copy(), self.rho_B.copy()))

            if n == nsteps:
                break
            self.step(dt, add_noise=add_noise)

        out = {
            "time": np.array(times),
            "mass_A": np.array(masses_A),
            "mass_B": np.array(masses_B),
            "mass_occupied": np.array(masses_occ),
            "mass_total": np.array(masses_tot),
            "rho_A": self.rho_A.copy(),
            "rho_B": self.rho_B.copy(),
        }
        if snapshot_every > 0:
            out["snapshots"] = np.array(snapshots, dtype=object)
        return out


def make_random_initial_condition(
    nx: int,
    ny: int,
    rhoA0: float = 0.35,
    rhoB0: float = 0.35,
    noise: float = 1e-2,
    seed: int = 0,
) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    rho_A = rhoA0 + noise * rng.standard_normal((ny, nx))
    rho_B = rhoB0 + noise * rng.standard_normal((ny, nx))
    return rho_A, rho_B


def main() -> None:
    nx, ny = 64, 64
    lx, ly = 1.0, 1.0
    dt = 5e-5
    nsteps = 200

    params = ModelParameters(
        D_A=0.1,
        D_B=0.1,
        D_v=0.05,
        beta=10.0,
        h_noise=min(lx / nx, ly / ny),
        kappa=np.array([[0.6, -0.2], [-0.2, 0.6]], dtype=float),
        gamma=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )

    solver = SchellingVoterFVSolverOptimized(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        params=params,
        bc="periodic",
        semi_implicit_stiff=True,
        stochastic=True,
        rng=np.random.default_rng(1234),
    )

    rho_A0, rho_B0 = make_random_initial_condition(nx, ny, seed=1)
    solver.set_state(rho_A0, rho_B0)

    m0 = solver.total_masses()
    result = solver.run(dt=dt, nsteps=nsteps, snapshot_every=0, add_noise=True)
    m1 = solver.total_masses()

    print("Initial masses:", m0)
    print("Final masses  :", m1)
    print("Mass drifts   :")
    print("  dM_A        =", m1["A"] - m0["A"])
    print("  dM_B        =", m1["B"] - m0["B"])
    print("  dM_occ      =", m1["occupied"] - m0["occupied"])
    print("  dM_total    =", m1["total"] - m0["total"])
    print("rho_A range   =", float(np.min(result["rho_A"])), float(np.max(result["rho_A"])))
    print("rho_B range   =", float(np.min(result["rho_B"])), float(np.max(result["rho_B"])))
    print(
        "rho_0 range   =",
        float(np.min(1.0 - result["rho_A"] - result["rho_B"])),
        float(np.max(1.0 - result["rho_A"] - result["rho_B"])),
    )


if __name__ == "__main__":
    main()
