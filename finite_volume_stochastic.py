#!/usr/bin/env python3
"""
Conservative Cartesian finite-volume solver for stochastic Schelling--Voter
hydrodynamics on a regular two-dimensional grid.

The solver evolves two occupied-density fields, ``rho_A`` and ``rho_B``.  The
vacancy field is defined by ``rho_0 = 1 - rho_A - rho_B``, so physically
admissible states lie in the cellwise simplex

    rho_A >= 0, rho_B >= 0, rho_A + rho_B <= 1.

Limiter modes are intentionally limited to ``limiter_mode="none"`` for raw
updates and ``limiter_mode="clip"`` for the historical local simplex
projection.

Numerical scheme
----------------
* The deterministic finite-volume update is conservative and face based.  It
  supports active masks and either periodic or no-flux boundary conditions.
* Conservative Schelling noise uses one Gaussian draw per face shared by the two
  adjacent cells, making the discrete stochastic flux exactly conservative.
* Voter demographic noise uses one Gaussian draw per cell with opposite signs
  for species A and B, preserving occupied mass up to floating-point roundoff.
* The optional semi-implicit Gamma step is evaluated in Fourier space.  It is
  currently available for periodic boundary conditions on a fully active grid.

This module is intended as a compact research implementation: it favors clear
finite-volume structure, reusable scratch buffers, and explicit documentation of
where conservation and limiting enter the update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time

import fhd
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
def _apply_simplex_limiter_numba(rho_A: Array, rho_B: Array, active: Array) -> int:
    """Project active cells into rho_A >= 0, rho_B >= 0, rho_A + rho_B <= 1."""
    ny, nx = rho_A.shape
    corrected = 0
    for iy in range(ny):
        for ix in range(nx):
            if not active[iy, ix]:
                if rho_A[iy, ix] != 0.0 or rho_B[iy, ix] != 0.0:
                    corrected += 1
                rho_A[iy, ix] = 0.0
                rho_B[iy, ix] = 0.0
                continue

            ra = rho_A[iy, ix]
            rb = rho_B[iy, ix]
            original_ra = ra
            original_rb = rb

            if not np.isfinite(ra):
                ra = 0.0
            if not np.isfinite(rb):
                rb = 0.0

            if ra < 0.0:
                ra = 0.0
            if rb < 0.0:
                rb = 0.0

            occupied = ra + rb
            if occupied > 1.0:
                inv_occupied = 1.0 / occupied
                ra *= inv_occupied
                rb *= inv_occupied

            rho_A[iy, ix] = ra
            rho_B[iy, ix] = rb
            if ra != original_ra or rb != original_rb:
                corrected += 1

    return corrected


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

    # rho0 is supplied by step()/explicit_rhs() so it is computed once per step.
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

    # rho0 is supplied by step()/stochastic_increment(); avoid recomputing it here.
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


@njit(cache=True)
def _compute_rho0_periodic_inplace(rho_A: Array, rho_B: Array, rho0: Array) -> None:
    """Fully-active shortcut: no mask checks while forming vacancies."""
    ny, nx = rho_A.shape
    for iy in range(ny):
        for ix in range(nx):
            rho0[iy, ix] = 1.0 - rho_A[iy, ix] - rho_B[iy, ix]


@njit(cache=True)
def _compute_linear_utility_periodic_inplace(
    rho_A: Array,
    rho_B: Array,
    k00: float,
    k01: float,
    k10: float,
    k11: float,
    U_A: Array,
    U_B: Array,
) -> None:
    """Fully-active shortcut for the local utility matrix multiply."""
    ny, nx = rho_A.shape
    for iy in range(ny):
        for ix in range(nx):
            ra = rho_A[iy, ix]
            rb = rho_B[iy, ix]
            U_A[iy, ix] = k00 * ra + k01 * rb
            U_B[iy, ix] = k10 * ra + k11 * rb


@njit(cache=True)
def _explicit_rhs_periodic_numba(
    rho_A: Array,
    rho_B: Array,
    dx: float,
    dy: float,
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
    """Fast conservative RHS for the common fully-active periodic grid.

    This is algebraically the same face update as the generic kernel, but it
    removes mask tests and boundary-condition branches from the hot path.
    """
    ny, nx = rho_A.shape
    _compute_linear_utility_periodic_inplace(rho_A, rho_B, k00, k01, k10, k11, U_A, U_B)
    _zero_array(rhs_A)
    _zero_array(rhs_B)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                jx = 0

            rhoA_x = 0.5 * (rho_A[iy, ix] + rho_A[iy, jx])
            rhoB_x = 0.5 * (rho_B[iy, ix] + rho_B[iy, jx])
            rho0_x = 0.5 * (rho0[iy, ix] + rho0[iy, jx])

            d_rhoA_x = (rho_A[iy, jx] - rho_A[iy, ix]) * inv_dx
            d_rhoB_x = (rho_B[iy, jx] - rho_B[iy, ix]) * inv_dx
            d_rho0_x = (rho0[iy, jx] - rho0[iy, ix]) * inv_dx
            d_UA_x = (U_A[iy, jx] - U_A[iy, ix]) * inv_dx
            d_UB_x = (U_B[iy, jx] - U_B[iy, ix]) * inv_dx

            F_A = D_A * (rho0_x * d_rhoA_x - rhoA_x * d_rho0_x - beta * rho0_x * rhoA_x * d_UA_x)
            F_A += D_v * (rhoB_x * d_rhoA_x - rhoA_x * d_rhoB_x)
            F_B = D_B * (rho0_x * d_rhoB_x - rhoB_x * d_rho0_x - beta * rho0_x * rhoB_x * d_UB_x)
            F_B += D_v * (rhoA_x * d_rhoB_x - rhoB_x * d_rhoA_x)

            contrib_A = F_A * inv_dx
            contrib_B = F_B * inv_dx
            rhs_A[iy, ix] += contrib_A
            rhs_A[iy, jx] -= contrib_A
            rhs_B[iy, ix] += contrib_B
            rhs_B[iy, jx] -= contrib_B

    for iy in range(ny):
        ky = iy + 1
        if ky == ny:
            ky = 0
        for ix in range(nx):
            rhoA_y = 0.5 * (rho_A[iy, ix] + rho_A[ky, ix])
            rhoB_y = 0.5 * (rho_B[iy, ix] + rho_B[ky, ix])
            rho0_y = 0.5 * (rho0[iy, ix] + rho0[ky, ix])

            d_rhoA_y = (rho_A[ky, ix] - rho_A[iy, ix]) * inv_dy
            d_rhoB_y = (rho_B[ky, ix] - rho_B[iy, ix]) * inv_dy
            d_rho0_y = (rho0[ky, ix] - rho0[iy, ix]) * inv_dy
            d_UA_y = (U_A[ky, ix] - U_A[iy, ix]) * inv_dy
            d_UB_y = (U_B[ky, ix] - U_B[iy, ix]) * inv_dy

            F_A = D_A * (rho0_y * d_rhoA_y - rhoA_y * d_rho0_y - beta * rho0_y * rhoA_y * d_UA_y)
            F_A += D_v * (rhoB_y * d_rhoA_y - rhoA_y * d_rhoB_y)
            F_B = D_B * (rho0_y * d_rhoB_y - rhoB_y * d_rho0_y - beta * rho0_y * rhoB_y * d_UB_y)
            F_B += D_v * (rhoA_y * d_rhoB_y - rhoB_y * d_rhoA_y)

            contrib_A = F_A * inv_dy
            contrib_B = F_B * inv_dy
            rhs_A[iy, ix] += contrib_A
            rhs_A[ky, ix] -= contrib_A
            rhs_B[iy, ix] += contrib_B
            rhs_B[ky, ix] -= contrib_B


@njit(cache=True)
def _stochastic_increment_periodic_numba(
    rho_A: Array,
    rho_B: Array,
    dx: float,
    dy: float,
    vol: float,
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
    """Fast stochastic increment for fully-active periodic grids."""
    ny, nx = rho_A.shape
    _zero_array(dW_A)
    _zero_array(dW_B)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    scale_cv = np.sqrt(dt / vol)

    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                jx = 0
            rhoA_x = 0.5 * (rho_A[iy, ix] + rho_A[iy, jx])
            rhoB_x = 0.5 * (rho_B[iy, ix] + rho_B[iy, jx])
            rho0_x = 0.5 * (rho0[iy, ix] + rho0[iy, jx])
            amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_x * rhoA_x, 0.0))
            amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_x * rhoB_x, 0.0))
            contrib_A = scale_cv * amp_A * eta_A_x[iy, ix] * inv_dx
            contrib_B = scale_cv * amp_B * eta_B_x[iy, ix] * inv_dx
            dW_A[iy, ix] += contrib_A
            dW_A[iy, jx] -= contrib_A
            dW_B[iy, ix] += contrib_B
            dW_B[iy, jx] -= contrib_B

    for iy in range(ny):
        ky = iy + 1
        if ky == ny:
            ky = 0
        for ix in range(nx):
            rhoA_y = 0.5 * (rho_A[iy, ix] + rho_A[ky, ix])
            rhoB_y = 0.5 * (rho_B[iy, ix] + rho_B[ky, ix])
            rho0_y = 0.5 * (rho0[iy, ix] + rho0[ky, ix])
            amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_y * rhoA_y, 0.0))
            amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_y * rhoB_y, 0.0))
            contrib_A = scale_cv * amp_A * eta_A_y[iy, ix] * inv_dy
            contrib_B = scale_cv * amp_B * eta_B_y[iy, ix] * inv_dy
            dW_A[iy, ix] += contrib_A
            dW_A[ky, ix] -= contrib_A
            dW_B[iy, ix] += contrib_B
            dW_B[ky, ix] -= contrib_B

    voter_prefactor = 2.0 * spatial_dim * D_v * (h_noise ** (spatial_dim - 2))
    for iy in range(ny):
        for ix in range(nx):
            amp_v = np.sqrt(max(voter_prefactor * rho_A[iy, ix] * rho_B[iy, ix], 0.0))
            source = scale_cv * amp_v * xi[iy, ix]
            dW_A[iy, ix] += source
            dW_B[iy, ix] -= source


# Explicit and stochastic kernels above keep separate generic and fully-active
# periodic implementations so common periodic runs avoid mask/boundary branches.

@dataclass
class ModelParameters:
    """Physical and numerical parameters for the finite-volume model.

    ``D_A`` and ``D_B`` set species-vacancy mobility scales, ``D_v`` controls
    voter exchange, ``beta`` scales utility-driven drift, and ``h_noise`` is the
    microscopic length scale entering the fluctuation amplitudes.  ``kappa`` is
    the local linear utility matrix, while ``gamma`` is the matrix used by the
    optional stiff nonlocal regularization step.
    """

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
            self.kappa = np.array([[0.6, -0.4], [-0.4, 0.6]], dtype=float)
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


class SchellingVoterFVSolver:
    """Finite-volume integrator for two-species Schelling--Voter fields.

    Parameters
    ----------
    nx, ny:
        Number of finite-volume cells in the x and y directions.
    lx, ly:
        Physical domain lengths.
    params:
        Transport, interaction, and stochastic-noise parameters.
    active:
        Optional boolean mask selecting cells that participate in the update.
        Inactive cells are forced to zero density by the simplex limiter.
    bc:
        Boundary condition for explicit fluxes, either ``"periodic"`` or
        ``"noflux"``.
    semi_implicit_stiff:
        If true, apply the Fourier-space semi-implicit Gamma step after the
        explicit/stochastic update.
    stochastic:
        Default value used by :meth:`step` and :meth:`run` when ``add_noise`` is
        not provided.
    simplex_limiter:
        Backward-compatible switch for the historical clip projection. Ignored
        when ``limiter_mode`` is supplied explicitly.
    limiter_mode:
        Either ``"none"`` or ``"clip"``. Removed experimental limiter modes are
        rejected rather than silently aliased.
    rng:
        Optional NumPy random generator used for stochastic increments.
    use_periodic_fast_path:
        If true, automatically dispatch fully active periodic grids to branch-free
        specialized kernels; disable for benchmarking the generic path.
    """

    @staticmethod
    def _empty_limiter_stats(mode: str) -> Dict[str, int | float | str]:
        """Return zero-valued diagnostics for the active limiter mode."""
        return {
            "mode": mode,
            "limited_cells": 0,
            "activations": 0,
            "residual_max_violation": 0.0,
            "projection_delta_A": 0.0,
            "projection_delta_B": 0.0,
            "projection_delta_occupied": 0.0,
            "projection_delta_total": 0.0,
        }

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
        simplex_limiter: bool = True,
        limiter_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        use_periodic_fast_path: bool = True,
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
        if limiter_mode is None:
            limiter_mode = "clip" if simplex_limiter else "none"
        self.limiter_mode = str(limiter_mode).lower()
        if self.limiter_mode not in {"none", "clip"}:
            raise ValueError("limiter_mode must be 'none' or 'clip'")
        self.simplex_limiter = self.limiter_mode == "clip"
        self.last_limiter_corrections = 0
        self.last_limiter_stats: Dict[str, int | float | str] = self._empty_limiter_stats(self.limiter_mode)
        self.rng = np.random.default_rng() if rng is None else rng
        self.use_periodic_fast_path = bool(use_periodic_fast_path)

        if self.bc not in {"periodic", "noflux"}:
            raise ValueError("bc must be 'periodic' or 'noflux'")

        if active is None:
            self.active = np.ones((self.ny, self.nx), dtype=np.bool_)
        else:
            active = np.asarray(active, dtype=np.bool_)
            if active.shape != (self.ny, self.nx):
                raise ValueError("active mask must have shape (ny, nx)")
            self.active = active.copy()

        self.fully_active_periodic = self.bc_periodic == 1 and bool(np.all(self.active))
        self._use_fast_periodic = self.use_periodic_fast_path and self.fully_active_periodic

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

        # Precompute real-FFT wave numbers and spectral work buffers for the
        # stiff step. rfft2 keeps only nx//2+1 columns for real-valued fields.
        kx = 2.0 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self._KX_r, self._KY_r = np.meshgrid(kx, ky)
        self._k2_r = self._KX_r**2 + self._KY_r**2
        self._k4_r = self._k2_r**2
        rshape = (self.ny, self.nx // 2 + 1)
        self._rhoA_hat = np.zeros(rshape, dtype=np.complex128)
        self._rhoB_hat = np.zeros(rshape, dtype=np.complex128)
        self._stiff_tmp_hat = np.zeros(rshape, dtype=np.complex128)
        self._stiff_det = np.zeros(rshape, dtype=np.float64)

    def apply_simplex_limiter(
        self,
        rho_A: Optional[Array] = None,
        rho_B: Optional[Array] = None,
    ) -> int:
        """Project density fields into the pointwise physical simplex.

        Parameters
        ----------
        rho_A, rho_B:
            Optional arrays to limit in-place.  When omitted, the solver state
            arrays are limited.

        Returns
        -------
        int
            Number of cells whose values were modified.
        """
        if (rho_A is None) != (rho_B is None):
            raise ValueError("rho_A and rho_B must be provided together")
        if rho_A is None:
            rho_A = self.rho_A
            rho_B = self.rho_B
        if rho_A.shape != (self.ny, self.nx) or rho_B.shape != (self.ny, self.nx):
            raise ValueError("rho_A and rho_B must have shape (ny, nx)")
        corrections = _apply_simplex_limiter_numba(rho_A, rho_B, self.active)
        self.last_limiter_corrections = int(corrections)
        return self.last_limiter_corrections

    def set_state(self, rho_A: Array, rho_B: Array) -> None:
        """Set the solver state and enforce mask/simplex constraints if enabled."""
        rho_A = np.asarray(rho_A, dtype=np.float64)
        rho_B = np.asarray(rho_B, dtype=np.float64)
        if rho_A.shape != (self.ny, self.nx) or rho_B.shape != (self.ny, self.nx):
            raise ValueError("rho_A and rho_B must have shape (ny, nx)")
        self.rho_A[...] = np.where(self.active, rho_A, 0.0)
        self.rho_B[...] = np.where(self.active, rho_B, 0.0)
        if self.limiter_mode == "clip":
            self.apply_simplex_limiter()

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

    def _compute_rho0_for_state(self, rho_A: Array, rho_B: Array) -> Array:
        """Refresh the vacancy scratch once for the supplied state."""
        if self._use_fast_periodic:
            _compute_rho0_periodic_inplace(rho_A, rho_B, self._rho0)
        else:
            _compute_rho0_inplace(rho_A, rho_B, self.active, self._rho0)
        return self._rho0

    def _fill_standard_normal(self, out: Array) -> None:
        """Fill RNG scratch in-place when NumPy supports Generator(..., out=...)."""
        try:
            self.rng.standard_normal(out=out)
        except TypeError:  # NumPy < 1.17-compatible fallback: same distribution, one temporary.
            out[...] = self.rng.standard_normal(out.shape)

    def explicit_rhs(self, rho_A: Array, rho_B: Array, rho0: Optional[Array] = None) -> Tuple[Array, Array]:
        if rho0 is None:
            rho0 = self._compute_rho0_for_state(rho_A, rho_B)

        if self._use_fast_periodic:
            _explicit_rhs_periodic_numba(
                rho_A,
                rho_B,
                self.dx,
                self.dy,
                self.params.D_A,
                self.params.D_B,
                self.params.D_v,
                self.params.beta,
                self.params.kappa[0, 0],
                self.params.kappa[0, 1],
                self.params.kappa[1, 0],
                self.params.kappa[1, 1],
                rho0,
                self._U_A,
                self._U_B,
                self._rhs_A,
                self._rhs_B,
            )
        else:
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
                rho0,
                self._U_A,
                self._U_B,
                self._rhs_A,
                self._rhs_B,
            )
        return self._rhs_A, self._rhs_B

    def stochastic_increment(
        self,
        rho_A: Array,
        rho_B: Array,
        dt: float,
        rho0: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        if rho0 is None:
            rho0 = self._compute_rho0_for_state(rho_A, rho_B)

        # Fill preallocated random buffers directly when supported; this avoids
        # five large temporary arrays per stochastic step on modern NumPy.
        self._fill_standard_normal(self._eta_A_x)
        self._fill_standard_normal(self._eta_B_x)
        self._fill_standard_normal(self._eta_A_y)
        self._fill_standard_normal(self._eta_B_y)
        self._fill_standard_normal(self._xi)

        if self._use_fast_periodic:
            _stochastic_increment_periodic_numba(
                rho_A,
                rho_B,
                self.dx,
                self.dy,
                self.vol,
                self.params.D_A,
                self.params.D_B,
                self.params.D_v,
                self.params.h_noise,
                self.params.spatial_dim,
                dt,
                rho0,
                self._dW_A,
                self._dW_B,
                self._eta_A_x,
                self._eta_B_x,
                self._eta_A_y,
                self._eta_B_y,
                self._xi,
            )
        else:
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
                rho0,
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

        # Reuse the vacancy scratch for the mobility averages used by the same
        # frozen-coefficient analytic 2x2 Fourier solve as before.
        self._rho0[...] = 1.0
        self._rho0 -= rho_A_star
        self._rho0 -= rho_B_star
        np.multiply(self._rho0, rho_A_star, out=self._rho_A_new)
        Mbar_A = self.params.beta * self.params.D_A * float(np.mean(self._rho_A_new))
        np.multiply(self._rho0, rho_B_star, out=self._rho_B_new)
        Mbar_B = self.params.beta * self.params.D_B * float(np.mean(self._rho_B_new))

        g = self.params.gamma
        alpha = dt * self._k4_r

        # Build the determinant in a reusable real scratch array. The m_ij arrays
        # are kept as scalar+alpha expressions to preserve the vectorized analytic
        # inversion while reducing persistent allocations.
        m11 = 1.0 + alpha * (Mbar_A * g[0, 0])
        m12 = alpha * (Mbar_A * g[0, 1])
        m21 = alpha * (Mbar_B * g[1, 0])
        m22 = 1.0 + alpha * (Mbar_B * g[1, 1])
        np.multiply(m11, m22, out=self._stiff_det)
        self._stiff_det -= m12 * m21

        self._rhoA_hat[...] = np.fft.rfft2(rho_A_star)
        self._rhoB_hat[...] = np.fft.rfft2(rho_B_star)
        rhoA_zero = self._rhoA_hat[0, 0]
        rhoB_zero = self._rhoB_hat[0, 0]

        # In-place 2x2 inverse: tmp = A_new, rhoB_hat = B_new, rhoA_hat = tmp.
        np.multiply(m22, self._rhoA_hat, out=self._stiff_tmp_hat)
        self._stiff_tmp_hat -= m12 * self._rhoB_hat
        self._stiff_tmp_hat /= self._stiff_det

        self._rhoB_hat *= m11
        self._rhoB_hat += (-m21) * self._rhoA_hat
        self._rhoB_hat /= self._stiff_det
        self._rhoA_hat[...] = self._stiff_tmp_hat

        # Preserve the exact zero mode => exact conservation of species masses in
        # the stiff step, matching the complex FFT implementation.
        self._rhoA_hat[0, 0] = rhoA_zero
        self._rhoB_hat[0, 0] = rhoB_zero

        self._rho_A_new[...] = np.fft.irfft2(self._rhoA_hat, s=(self.ny, self.nx))
        self._rho_B_new[...] = np.fft.irfft2(self._rhoB_hat, s=(self.ny, self.nx))
        return self._rho_A_new, self._rho_B_new

    def _max_simplex_violation(self) -> float:
        return float(
            max(
                0.0,
                -float(np.min(self.rho_A)),
                -float(np.min(self.rho_B)),
                float(np.max(self.rho_A + self.rho_B - 1.0)),
            )
        )

    def step(self, dt: float, add_noise: Optional[bool] = None) -> None:
        if add_noise is None:
            add_noise = self.stochastic

        # Compute vacancy once for this timestep and share it between the
        # deterministic and stochastic kernels. Both increments are evaluated at
        # the same pre-step state, preserving the original scheme.
        rho0 = self._compute_rho0_for_state(self.rho_A, self.rho_B)
        rhs_A, rhs_B = self.explicit_rhs(self.rho_A, self.rho_B, rho0=rho0)
        self._rho_A_star[...] = self.rho_A + dt * rhs_A
        self._rho_B_star[...] = self.rho_B + dt * rhs_B

        if add_noise:
            dW_A, dW_B = self.stochastic_increment(self.rho_A, self.rho_B, dt, rho0=rho0)
            self._rho_A_star += dW_A
            self._rho_B_star += dW_B

        limiter_corrections = 0
        projection_delta_A = 0.0
        projection_delta_B = 0.0
        projection_delta_occupied = 0.0
        if self.limiter_mode == "clip":
            before_A = float(np.sum(self._rho_A_star[self.active]) * self.vol)
            before_B = float(np.sum(self._rho_B_star[self.active]) * self.vol)
            limiter_corrections += self.apply_simplex_limiter(self._rho_A_star, self._rho_B_star)
            projection_delta_A += float(np.sum(self._rho_A_star[self.active]) * self.vol) - before_A
            projection_delta_B += float(np.sum(self._rho_B_star[self.active]) * self.vol) - before_B

        if self.semi_implicit_stiff:
            rho_A_new, rho_B_new = self.semi_implicit_gamma_step(self._rho_A_star, self._rho_B_star, dt)
        else:
            rho_A_new = self._rho_A_star
            rho_B_new = self._rho_B_star

        if self.fully_active_periodic:
            self.rho_A[...] = rho_A_new
            self.rho_B[...] = rho_B_new
        else:
            self.rho_A[...] = np.where(self.active, rho_A_new, 0.0)
            self.rho_B[...] = np.where(self.active, rho_B_new, 0.0)
        if self.limiter_mode == "clip":
            before_A = float(np.sum(self.rho_A[self.active]) * self.vol)
            before_B = float(np.sum(self.rho_B[self.active]) * self.vol)
            limiter_corrections += self.apply_simplex_limiter()
            projection_delta_A += float(np.sum(self.rho_A[self.active]) * self.vol) - before_A
            projection_delta_B += float(np.sum(self.rho_B[self.active]) * self.vol) - before_B
            projection_delta_occupied = projection_delta_A + projection_delta_B
            self.last_limiter_corrections = limiter_corrections
            self.last_limiter_stats = self._empty_limiter_stats(self.limiter_mode)
            self.last_limiter_stats.update(
                {
                    "limited_cells": int(limiter_corrections),
                    "activations": 1 if limiter_corrections > 0 else 0,
                    "residual_max_violation": self._max_simplex_violation(),
                    "projection_delta_A": projection_delta_A,
                    "projection_delta_B": projection_delta_B,
                    "projection_delta_occupied": projection_delta_occupied,
                    "projection_delta_total": 0.0,
                }
            )
        else:
            self.last_limiter_corrections = 0
            self.last_limiter_stats = self._empty_limiter_stats(self.limiter_mode)
            self.last_limiter_stats["residual_max_violation"] = self._max_simplex_violation()

    def run(
        self,
        dt: float,
        nsteps: int,
        snapshot_every: int = 0,
        add_noise: Optional[bool] = None,
        record_masses_every: int = 0,
        record_snapshots_every: Optional[int] = None,
        verbatum: Optional[bool] = False
    ) -> Dict[str, Array]:
        """Advance for ``nsteps`` with optional diagnostics decimation.

        ``record_masses_every=1`` reproduces the previous behavior of computing
        mass diagnostics at every step. The default ``0`` records only the
        initial and final masses, avoiding an O(ncells) diagnostic pass per step
        in production runs. ``record_snapshots_every`` is an alias for the older
        ``snapshot_every`` argument; when omitted, ``snapshot_every`` is used.
        """
        if record_snapshots_every is None:
            record_snapshots_every = snapshot_every
            record_masses_every = snapshot_every
        if record_masses_every < 0 or record_snapshots_every < 0:
            raise ValueError("record intervals must be non-negative")

        masses_A = []
        masses_B = []
        masses_occ = []
        masses_tot = []
        mass_times = []
        limiter_limited_cells = []
        limiter_activations = []
        limiter_residual_max_violation = []
        limiter_projection_delta_A = []
        limiter_projection_delta_B = []
        limiter_projection_delta_occupied = []
        limiter_projection_delta_total = []
        snapshots = []

        def should_record_mass(step_index: int) -> bool:
            if step_index == 0 or step_index == nsteps:
                return True
            return record_masses_every > 0 and step_index % record_masses_every == 0

        for n in range(nsteps + 1):
            if should_record_mass(n):
                m = self.total_masses()
                masses_A.append(m["A"])
                masses_B.append(m["B"])
                masses_occ.append(m["occupied"])
                masses_tot.append(m["total"])
                mass_times.append(n * dt)
                limiter_limited_cells.append(int(self.last_limiter_stats.get("limited_cells", 0)))
                limiter_activations.append(int(self.last_limiter_stats.get("activations", 0)))
                limiter_residual_max_violation.append(float(self.last_limiter_stats.get("residual_max_violation", 0.0)))
                limiter_projection_delta_A.append(float(self.last_limiter_stats.get("projection_delta_A", 0.0)))
                limiter_projection_delta_B.append(float(self.last_limiter_stats.get("projection_delta_B", 0.0)))
                limiter_projection_delta_occupied.append(float(self.last_limiter_stats.get("projection_delta_occupied", 0.0)))
                limiter_projection_delta_total.append(float(self.last_limiter_stats.get("projection_delta_total", 0.0)))
                # fhd diagnostics expect species-first fields with shape (2, nx, ny).
                phi = np.array([self.rho_A.T, self.rho_B.T])
                if verbatum:
                    dissimilarity_index = fhd.dissimilarity(phi)
                    mean_kl_divergence = fhd.mean_relative_entropy(phi)
                    print(
                        f"step {n}/{nsteps}:    "
                        f"<m_A> = {masses_A[-1]/masses_tot[-1]:.6f},   "
                        f"<m_B> = {masses_B[-1]/masses_tot[-1]:.6f},    "
                        f"<m_occ> = {masses_occ[-1]/masses_tot[-1]:.6f},    "
                        f"D_index = {dissimilarity_index:.6f},    "
                        f"mean_kl_divergence = {mean_kl_divergence:.6f}"
                    )

            if record_snapshots_every > 0 and n % record_snapshots_every == 0:
                snapshots.append((n * dt, self.rho_A.copy(), self.rho_B.copy()))

            if n == nsteps:
                break
            self.step(dt, add_noise=add_noise)

        out = {
            "time": np.array(mass_times),
            "mass_A": np.array(masses_A),
            "mass_B": np.array(masses_B),
            "mass_occupied": np.array(masses_occ),
            "mass_total": np.array(masses_tot),
            "rho_A": self.rho_A.copy(),
            "rho_B": self.rho_B.copy(),
            "limiter_mode": self.limiter_mode,
            "limiter_limited_cells": np.array(limiter_limited_cells),
            "limiter_activations": np.array(limiter_activations),
            "limiter_residual_max_violation": np.array(limiter_residual_max_violation),
            "limiter_projection_delta_A": np.array(limiter_projection_delta_A),
            "limiter_projection_delta_B": np.array(limiter_projection_delta_B),
            "limiter_projection_delta_occupied": np.array(limiter_projection_delta_occupied),
            "limiter_projection_delta_total": np.array(limiter_projection_delta_total),
        }
        if record_snapshots_every > 0:
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
    """Create a noisy two-species initial condition on a ``(ny, nx)`` grid.

    The returned arrays are intentionally not clipped here so callers can choose
    whether to inspect raw perturbations or rely on ``SchellingVoterFVSolver`` to
    apply its simplex limiter during ``set_state``.
    """
    rng = np.random.default_rng(seed)
    rho_A = rhoA0 + noise * rng.standard_normal((ny, nx))
    rho_B = rhoB0 + noise * rng.standard_normal((ny, nx))
    return rho_A, rho_B


def benchmark_fast_paths(nx: int = 96, ny: int = 96, nsteps: int = 100, dt: float = 2e-5) -> None:
    """Compare generic masked periodic and specialized periodic kernels.

    The generic case disables only the fast-path dispatcher while keeping the
    same fully active periodic state, so timings isolate the hot-kernel speedup.
    """
    lx = ly = 1.0
    params = ModelParameters(
        D_A=0.1,
        D_B=0.1,
        D_v=0.05,
        beta=10.0,
        h_noise=min(lx / nx, ly / ny),
        kappa=np.array([[0.6, -0.4], [-0.4, 0.6]], dtype=float),
        gamma=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )
    rho_A0, rho_B0 = make_random_initial_condition(nx, ny, seed=11)

    def build_solver(seed: int, use_fast: bool, stochastic: bool) -> SchellingVoterFVSolver:
        solver = SchellingVoterFVSolver(
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            params=params,
            bc="periodic",
            semi_implicit_stiff=True,
            stochastic=stochastic,
            simplex_limiter=False,
            rng=np.random.default_rng(seed),
            use_periodic_fast_path=use_fast,
        )
        solver.set_state(rho_A0, rho_B0)
        return solver

    # Trigger Numba compilation outside the timed region for both dispatch paths.
    for use_fast in (False, True):
        for stochastic in (False, True):
            warm = build_solver(123, use_fast, stochastic)
            warm.step(dt, add_noise=stochastic)

    print(f"Benchmark: {nx}x{ny}, nsteps={nsteps}, dt={dt:g}")
    for stochastic in (False, True):
        label = "with stochastic noise" if stochastic else "deterministic only"
        timings = {}
        for name, use_fast in (("generic periodic", False), ("fast periodic", True)):
            solver = build_solver(123, use_fast, stochastic)
            t0 = time.perf_counter()
            solver.run(dt=dt, nsteps=nsteps, add_noise=stochastic, record_masses_every=0)
            elapsed = time.perf_counter() - t0
            timings[name] = elapsed
            masses = solver.total_masses()
            print(
                f"  {label:24s} | {name:16s}: {elapsed:8.4f} s "
                f"(M_occ={masses['occupied']:.12g}, M_total={masses['total']:.12g})"
            )
        speedup = timings["generic periodic"] / timings["fast periodic"]
        print(f"  speedup ({label}): {speedup:.3f}x")


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
        kappa=np.array([[0.6, -0.4], [-0.4, 0.6]], dtype=float),
        gamma=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )

    solver = SchellingVoterFVSolver(
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
