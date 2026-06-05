#!/usr/bin/env python3
"""
Conservative Cartesian finite-volume solver for stochastic Schelling--Voter
hydrodynamics on a regular two-dimensional grid.

The solver evolves two occupied-density fields, ``rho_A`` and ``rho_B``.  The
vacancy field is defined by ``rho_0 = 1 - rho_A - rho_B``, so physically
admissible states lie in the cellwise simplex

    rho_A >= 0, rho_B >= 0, rho_A + rho_B <= 1.

Three limiter modes are available.  ``limiter_mode="none"`` leaves raw
updates untouched, ``"clip"`` keeps the historical local simplex projection for
comparison, and ``"conservative"`` uses a face-flux budget limiter on the fully
active periodic Cartesian path.  The conservative limiter rescales shared face
increments before committing the update, so admissibility is enforced without the
occupied-mass drift caused by local post-step clipping.

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


@njit(cache=True)
def _conservative_limited_update_periodic_numba(
    rho_A: Array,
    rho_B: Array,
    dx: float,
    dy: float,
    vol: float,
    D_A: float,
    D_B: float,
    D_v: float,
    beta: float,
    h_noise: float,
    spatial_dim: int,
    k00: float,
    k01: float,
    k10: float,
    k11: float,
    dt: float,
    add_noise: int,
    rho0: Array,
    U_A: Array,
    U_B: Array,
    eta_A_x: Array,
    eta_B_x: Array,
    eta_A_y: Array,
    eta_B_y: Array,
    xi: Array,
    flux_A_x: Array,
    flux_B_x: Array,
    flux_A_y: Array,
    flux_B_y: Array,
    delta_A: Array,
    delta_B: Array,
    neg_A: Array,
    neg_B: Array,
    pos_occ: Array,
    alpha_A: Array,
    alpha_B: Array,
    alpha_occ: Array,
    rho_A_out: Array,
    rho_B_out: Array,
) -> Tuple[int, int, int]:
    """Bound-preserving conservative flux update for fully-active periodic grids.

    Face arrays store increments with the same sign convention as the RHS
    kernels: an x-face value is added to cell (iy, ix) and subtracted from its
    right neighbor; a y-face value is added to cell (iy, ix) and subtracted from
    its upward neighbor.  A single limiter coefficient is then applied to each
    shared face increment, so species masses and occupied mass are conserved by
    construction.  The voter source is handled after the flux step by clipping
    only the local A<->B exchange amount, which preserves occupied density.
    """
    ny, nx = rho_A.shape
    _compute_rho0_periodic_inplace(rho_A, rho_B, rho0)
    _compute_linear_utility_periodic_inplace(rho_A, rho_B, k00, k01, k10, k11, U_A, U_B)
    _zero_array(delta_A)
    _zero_array(delta_B)
    _zero_array(neg_A)
    _zero_array(neg_B)
    _zero_array(pos_occ)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    scale_cv = 0.0
    if add_noise == 1:
        scale_cv = np.sqrt(dt / vol)

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
            cA = dt * F_A * inv_dx
            cB = dt * F_B * inv_dx
            if add_noise == 1:
                amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_x * rhoA_x, 0.0))
                amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_x * rhoB_x, 0.0))
                cA += scale_cv * amp_A * eta_A_x[iy, ix] * inv_dx
                cB += scale_cv * amp_B * eta_B_x[iy, ix] * inv_dx
            flux_A_x[iy, ix] = cA
            flux_B_x[iy, ix] = cB
            delta_A[iy, ix] += cA
            delta_A[iy, jx] -= cA
            delta_B[iy, ix] += cB
            delta_B[iy, jx] -= cB

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
            cA = dt * F_A * inv_dy
            cB = dt * F_B * inv_dy
            if add_noise == 1:
                amp_A = h_noise * np.sqrt(max(2.0 * D_A * rho0_y * rhoA_y, 0.0))
                amp_B = h_noise * np.sqrt(max(2.0 * D_B * rho0_y * rhoB_y, 0.0))
                cA += scale_cv * amp_A * eta_A_y[iy, ix] * inv_dy
                cB += scale_cv * amp_B * eta_B_y[iy, ix] * inv_dy
            flux_A_y[iy, ix] = cA
            flux_B_y[iy, ix] = cB
            delta_A[iy, ix] += cA
            delta_A[ky, ix] -= cA
            delta_B[iy, ix] += cB
            delta_B[ky, ix] -= cB

    # Accumulate the raw negative-species and positive-occupancy demands from
    # each incident face contribution separately.  This makes the budget limiter
    # robust even when positive and negative face contributions cancel in a cell.
    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                jx = 0
            cA = flux_A_x[iy, ix]
            cB = flux_B_x[iy, ix]
            occ = cA + cB
            if cA < 0.0:
                neg_A[iy, ix] += -cA
            if cB < 0.0:
                neg_B[iy, ix] += -cB
            if occ > 0.0:
                pos_occ[iy, ix] += occ
            cA = -cA
            cB = -cB
            occ = cA + cB
            if cA < 0.0:
                neg_A[iy, jx] += -cA
            if cB < 0.0:
                neg_B[iy, jx] += -cB
            if occ > 0.0:
                pos_occ[iy, jx] += occ

            ky = iy + 1
            if ky == ny:
                ky = 0
            cA = flux_A_y[iy, ix]
            cB = flux_B_y[iy, ix]
            occ = cA + cB
            if cA < 0.0:
                neg_A[iy, ix] += -cA
            if cB < 0.0:
                neg_B[iy, ix] += -cB
            if occ > 0.0:
                pos_occ[iy, ix] += occ
            cA = -cA
            cB = -cB
            occ = cA + cB
            if cA < 0.0:
                neg_A[ky, ix] += -cA
            if cB < 0.0:
                neg_B[ky, ix] += -cB
            if occ > 0.0:
                pos_occ[ky, ix] += occ

    limited_cells = 0
    eps = 1.0e-15
    for iy in range(ny):
        for ix in range(nx):
            a = 1.0
            if neg_A[iy, ix] > rho_A[iy, ix] + eps:
                a = max(rho_A[iy, ix], 0.0) / neg_A[iy, ix]
            alpha_A[iy, ix] = a
            b = 1.0
            if neg_B[iy, ix] > rho_B[iy, ix] + eps:
                b = max(rho_B[iy, ix], 0.0) / neg_B[iy, ix]
            alpha_B[iy, ix] = b
            o = 1.0
            vacancy = 1.0 - rho_A[iy, ix] - rho_B[iy, ix]
            if pos_occ[iy, ix] > vacancy + eps:
                o = max(vacancy, 0.0) / pos_occ[iy, ix]
            alpha_occ[iy, ix] = o
            if a < 1.0 or b < 1.0 or o < 1.0:
                limited_cells += 1
            delta_A[iy, ix] = 0.0
            delta_B[iy, ix] = 0.0

    limited_faces = 0
    for iy in range(ny):
        for ix in range(nx):
            jx = ix + 1
            if jx == nx:
                jx = 0
            cA = flux_A_x[iy, ix]
            cB = flux_B_x[iy, ix]
            theta = 1.0
            occ = cA + cB
            if cA < 0.0 and alpha_A[iy, ix] < theta:
                theta = alpha_A[iy, ix]
            if cB < 0.0 and alpha_B[iy, ix] < theta:
                theta = alpha_B[iy, ix]
            if occ > 0.0 and alpha_occ[iy, ix] < theta:
                theta = alpha_occ[iy, ix]
            if cA > 0.0 and alpha_A[iy, jx] < theta:
                theta = alpha_A[iy, jx]
            if cB > 0.0 and alpha_B[iy, jx] < theta:
                theta = alpha_B[iy, jx]
            if occ < 0.0 and alpha_occ[iy, jx] < theta:
                theta = alpha_occ[iy, jx]
            if theta < 1.0:
                limited_faces += 1
            cA *= theta
            cB *= theta
            delta_A[iy, ix] += cA
            delta_A[iy, jx] -= cA
            delta_B[iy, ix] += cB
            delta_B[iy, jx] -= cB

    for iy in range(ny):
        ky = iy + 1
        if ky == ny:
            ky = 0
        for ix in range(nx):
            cA = flux_A_y[iy, ix]
            cB = flux_B_y[iy, ix]
            theta = 1.0
            occ = cA + cB
            if cA < 0.0 and alpha_A[iy, ix] < theta:
                theta = alpha_A[iy, ix]
            if cB < 0.0 and alpha_B[iy, ix] < theta:
                theta = alpha_B[iy, ix]
            if occ > 0.0 and alpha_occ[iy, ix] < theta:
                theta = alpha_occ[iy, ix]
            if cA > 0.0 and alpha_A[ky, ix] < theta:
                theta = alpha_A[ky, ix]
            if cB > 0.0 and alpha_B[ky, ix] < theta:
                theta = alpha_B[ky, ix]
            if occ < 0.0 and alpha_occ[ky, ix] < theta:
                theta = alpha_occ[ky, ix]
            if theta < 1.0:
                limited_faces += 1
            cA *= theta
            cB *= theta
            delta_A[iy, ix] += cA
            delta_A[ky, ix] -= cA
            delta_B[iy, ix] += cB
            delta_B[ky, ix] -= cB

    voter_limited_cells = 0
    voter_prefactor = 2.0 * spatial_dim * D_v * (h_noise ** (spatial_dim - 2))
    for iy in range(ny):
        for ix in range(nx):
            ra = rho_A[iy, ix] + delta_A[iy, ix]
            rb = rho_B[iy, ix] + delta_B[iy, ix]
            if add_noise == 1:
                amp_v = np.sqrt(max(voter_prefactor * rho_A[iy, ix] * rho_B[iy, ix], 0.0))
                source = scale_cv * amp_v * xi[iy, ix]
                lo = -ra
                hi = rb
                clipped = source
                if clipped < lo:
                    clipped = lo
                if clipped > hi:
                    clipped = hi
                if clipped != source:
                    voter_limited_cells += 1
                ra += clipped
                rb -= clipped
            # Tiny roundoff cleanup inside the simplex.  These assignments should
            # not change masses except at machine precision after the flux limits.
            if ra < 0.0 and ra > -1.0e-14:
                ra = 0.0
            if rb < 0.0 and rb > -1.0e-14:
                rb = 0.0
            occ = ra + rb
            if occ > 1.0 and occ < 1.0 + 1.0e-14:
                excess = occ - 1.0
                if ra >= rb:
                    ra -= excess
                else:
                    rb -= excess
            rho_A_out[iy, ix] = ra
            rho_B_out[iy, ix] = rb

    return limited_cells, limited_faces, voter_limited_cells

# Numba parallelization is intentionally not enabled here: the conservative
# face-scatter updates write to both neighboring cells, so naive prange would
# introduce races. A parallel two-pass face-flux implementation can be added
# later without changing the public API.


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
        Backward-compatible switch for the historical clip projection.  Ignored
        when ``limiter_mode`` is supplied explicitly.
    limiter_mode:
        One of ``"none"``, ``"clip"``, or ``"conservative"``.  The
        conservative mode is implemented for fully active periodic grids and
        limits explicit/stochastic face fluxes before committing the step.
    rng:
        Optional NumPy random generator used for stochastic increments.
    use_periodic_fast_path:
        If true, automatically dispatch fully active periodic grids to branch-free
        specialized kernels; disable for benchmarking the generic path.
    """

    @staticmethod
    def _empty_limiter_stats(mode: str) -> Dict[str, int | float | str]:
        """Return a complete zero-valued limiter diagnostics dictionary."""
        stats: Dict[str, int | float | str] = {
            "mode": mode,
            "limited_cells": 0,
            "limited_faces": 0,
            "limited_faces_x": 0,
            "limited_faces_y": 0,
            "voter_limited_cells": 0,
            "gamma_repair_cells": 0,
            "activations": 0,
            "residual_max_violation": 0.0,
            "theta_min": 1.0,
            "theta_mean": 1.0,
            "theta_median": 1.0,
            "theta_max": 1.0,
            "theta_x_min": 1.0,
            "theta_x_mean": 1.0,
            "theta_x_median": 1.0,
            "theta_x_max": 1.0,
            "theta_y_min": 1.0,
            "theta_y_mean": 1.0,
            "theta_y_median": 1.0,
            "theta_y_max": 1.0,
            "theta_frac_lt_1": 0.0,
            "theta_frac_lt_0_5": 0.0,
            "theta_frac_lt_0_1": 0.0,
            "theta_p01": 1.0,
            "theta_p05": 1.0,
            "theta_p10": 1.0,
            "theta_p25": 1.0,
            "theta_p75": 1.0,
            "theta_p90": 1.0,
            "theta_p95": 1.0,
            "theta_p99": 1.0,
        }
        for prefix in ("A", "B", "occupied", "deterministic_A", "deterministic_B", "deterministic_occupied", "schelling_noise_A", "schelling_noise_B", "schelling_noise_occupied"):
            stats[f"flux_{prefix}_l1_raw"] = 0.0
            stats[f"flux_{prefix}_l1_limited"] = 0.0
            stats[f"flux_{prefix}_l2_raw"] = 0.0
            stats[f"flux_{prefix}_l2_limited"] = 0.0
            stats[f"flux_{prefix}_removed_fraction"] = 0.0
        for name in ("rho_A", "rho_B", "rho_0", "grad_rho"):
            stats[f"limiting_corr_{name}"] = 0.0
        return stats

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
        if self.limiter_mode not in {"none", "clip", "conservative"}:
            raise ValueError("limiter_mode must be 'none', 'clip', or 'conservative'")
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

        # Face and budget scratch for the conservative bound-preserving limiter.
        # The limiter is slower than clipping because it must assemble face
        # increments, compute per-cell admissible flux budgets, and revisit each
        # shared face with a common coefficient.  This extra work preserves the
        # finite-volume conservation structure and is preferable in regimes where
        # stochastic kicks or sharp interfaces trigger frequent local projection;
        # the old clip projection can otherwise remove/add occupied mass cell by
        # cell and produce long-run mass drift.
        self._flux_A_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._flux_B_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._flux_A_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._flux_B_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._budget_A = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._budget_B = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._budget_occ = np.zeros((self.ny, self.nx), dtype=np.float64)
        self._alpha_A = np.ones((self.ny, self.nx), dtype=np.float64)
        self._alpha_B = np.ones((self.ny, self.nx), dtype=np.float64)
        self._alpha_occ = np.ones((self.ny, self.nx), dtype=np.float64)
        self._theta_x = np.ones((self.ny, self.nx), dtype=np.float64)
        self._theta_y = np.ones((self.ny, self.nx), dtype=np.float64)

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
        if self.limiter_mode in {"clip", "conservative"}:
            # Initial data are not produced by solver fluxes; use the historical
            # simplex projection to sanitize user-provided states in both bounded
            # modes.  Conservation claims below concern time stepping from an
            # admissible state.
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

    def _conservative_global_simplex_repair(
        self,
        rho_A: Array,
        rho_B: Array,
        target_mass_A: float,
        target_mass_B: float,
    ) -> Tuple[int, float]:
        """Conservative fallback repair for non-flux substeps such as Gamma.

        The long-term limiter is face based for FV/stochastic increments.  The
        Fourier Gamma solve has no local face flux representation, so if it
        creates small bound violations we repair them by redistributing species
        mass globally over fully active periodic cells while preserving the
        target species masses.  This path is intentionally slower and is counted
        as residual correction diagnostics rather than as face limiting.
        """
        rho_A[...] = np.where(np.isfinite(rho_A), rho_A, 0.0)
        rho_B[...] = np.where(np.isfinite(rho_B), rho_B, 0.0)
        np.maximum(rho_A, 0.0, out=rho_A)
        np.maximum(rho_B, 0.0, out=rho_B)
        occ = rho_A + rho_B
        over = occ > 1.0
        changed = int(np.count_nonzero(over))
        if np.any(over):
            scale = np.ones_like(occ)
            scale[over] = 1.0 / occ[over]
            rho_A *= scale
            rho_B *= scale

        target_sum_A = target_mass_A / self.vol
        target_sum_B = target_mass_B / self.vol
        eps = 1.0e-13
        for _ in range(64):
            diff_A = target_sum_A - float(np.sum(rho_A))
            if abs(diff_A) > eps:
                if diff_A > 0.0:
                    cap = 1.0 - rho_A - rho_B
                    total = float(np.sum(np.maximum(cap, 0.0)))
                    if total <= eps:
                        break
                    rho_A += np.maximum(cap, 0.0) * min(1.0, diff_A / total)
                else:
                    avail = float(np.sum(rho_A))
                    if avail <= eps:
                        break
                    rho_A *= max(0.0, 1.0 + diff_A / avail)
                changed += 1

            diff_B = target_sum_B - float(np.sum(rho_B))
            if abs(diff_B) > eps:
                if diff_B > 0.0:
                    cap = 1.0 - rho_A - rho_B
                    total = float(np.sum(np.maximum(cap, 0.0)))
                    if total <= eps:
                        break
                    rho_B += np.maximum(cap, 0.0) * min(1.0, diff_B / total)
                else:
                    avail = float(np.sum(rho_B))
                    if avail <= eps:
                        break
                    rho_B *= max(0.0, 1.0 + diff_B / avail)
                changed += 1

            violation = max(
                0.0,
                -float(np.min(rho_A)),
                -float(np.min(rho_B)),
                float(np.max(rho_A + rho_B - 1.0)),
                abs(target_sum_A - float(np.sum(rho_A))),
                abs(target_sum_B - float(np.sum(rho_B))),
            )
            if violation <= 5.0e-12:
                return changed, violation * self.vol
        residual = max(
            0.0,
            -float(np.min(rho_A)),
            -float(np.min(rho_B)),
            float(np.max(rho_A + rho_B - 1.0)),
            abs(target_sum_A - float(np.sum(rho_A))),
            abs(target_sum_B - float(np.sum(rho_B))),
        )
        return changed, residual * self.vol

    @staticmethod
    def _safe_corrcoef(x: Array, y: Array) -> float:
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        x_std = float(np.std(x_flat))
        y_std = float(np.std(y_flat))
        if x_std <= 0.0 or y_std <= 0.0:
            return 0.0
        return float(np.mean((x_flat - float(np.mean(x_flat))) * (y_flat - float(np.mean(y_flat)))) / (x_std * y_std))

    @staticmethod
    def _add_flux_norm_stats(stats: Dict[str, int | float | str], prefix: str, raw_x: Array, raw_y: Array, theta_x: Array, theta_y: Array) -> None:
        limited_x = theta_x * raw_x
        limited_y = theta_y * raw_y
        l1_raw = float(np.sum(np.abs(raw_x)) + np.sum(np.abs(raw_y)))
        l1_limited = float(np.sum(np.abs(limited_x)) + np.sum(np.abs(limited_y)))
        l2_raw = float(np.sqrt(np.sum(raw_x * raw_x) + np.sum(raw_y * raw_y)))
        l2_limited = float(np.sqrt(np.sum(limited_x * limited_x) + np.sum(limited_y * limited_y)))
        stats[f"flux_{prefix}_l1_raw"] = l1_raw
        stats[f"flux_{prefix}_l1_limited"] = l1_limited
        stats[f"flux_{prefix}_l2_raw"] = l2_raw
        stats[f"flux_{prefix}_l2_limited"] = l2_limited
        stats[f"flux_{prefix}_removed_fraction"] = 0.0 if l1_raw <= 0.0 else max(0.0, (l1_raw - l1_limited) / l1_raw)

    def _compute_theta_from_alpha(self) -> Tuple[Array, Array]:
        ax_r = np.roll(self._alpha_A, -1, axis=1)
        bx_r = np.roll(self._alpha_B, -1, axis=1)
        ox_r = np.roll(self._alpha_occ, -1, axis=1)
        ay_u = np.roll(self._alpha_A, -1, axis=0)
        by_u = np.roll(self._alpha_B, -1, axis=0)
        oy_u = np.roll(self._alpha_occ, -1, axis=0)

        np.copyto(self._theta_x, 1.0)
        cA = self._flux_A_x
        cB = self._flux_B_x
        occ = cA + cB
        np.minimum(self._theta_x, np.where(cA < 0.0, self._alpha_A, 1.0), out=self._theta_x)
        np.minimum(self._theta_x, np.where(cB < 0.0, self._alpha_B, 1.0), out=self._theta_x)
        np.minimum(self._theta_x, np.where(occ > 0.0, self._alpha_occ, 1.0), out=self._theta_x)
        np.minimum(self._theta_x, np.where(cA > 0.0, ax_r, 1.0), out=self._theta_x)
        np.minimum(self._theta_x, np.where(cB > 0.0, bx_r, 1.0), out=self._theta_x)
        np.minimum(self._theta_x, np.where(occ < 0.0, ox_r, 1.0), out=self._theta_x)

        np.copyto(self._theta_y, 1.0)
        cA = self._flux_A_y
        cB = self._flux_B_y
        occ = cA + cB
        np.minimum(self._theta_y, np.where(cA < 0.0, self._alpha_A, 1.0), out=self._theta_y)
        np.minimum(self._theta_y, np.where(cB < 0.0, self._alpha_B, 1.0), out=self._theta_y)
        np.minimum(self._theta_y, np.where(occ > 0.0, self._alpha_occ, 1.0), out=self._theta_y)
        np.minimum(self._theta_y, np.where(cA > 0.0, ay_u, 1.0), out=self._theta_y)
        np.minimum(self._theta_y, np.where(cB > 0.0, by_u, 1.0), out=self._theta_y)
        np.minimum(self._theta_y, np.where(occ < 0.0, oy_u, 1.0), out=self._theta_y)
        return self._theta_x, self._theta_y

    def _collect_conservative_limiter_diagnostics(
        self,
        dt: float,
        add_noise: bool,
        limited_cells: int,
        limited_faces: int,
        voter_limited_cells: int,
        gamma_repair_cells: int,
        gamma_repair_residual: float,
    ) -> Dict[str, int | float | str]:
        theta_x, theta_y = self._compute_theta_from_alpha()
        theta = np.concatenate((theta_x.ravel(), theta_y.ravel()))
        stats = self._empty_limiter_stats(self.limiter_mode)
        stats.update(
            {
                "limited_cells": int(limited_cells),
                "limited_faces": int(limited_faces),
                "limited_faces_x": int(np.count_nonzero(theta_x < 1.0)),
                "limited_faces_y": int(np.count_nonzero(theta_y < 1.0)),
                "voter_limited_cells": int(voter_limited_cells),
                "gamma_repair_cells": int(gamma_repair_cells),
                "activations": 1 if (limited_cells > 0 or limited_faces > 0 or voter_limited_cells > 0 or gamma_repair_cells > 0) else 0,
                "residual_max_violation": max(self._max_simplex_violation(), gamma_repair_residual),
                "theta_min": float(np.min(theta)),
                "theta_mean": float(np.mean(theta)),
                "theta_median": float(np.median(theta)),
                "theta_max": float(np.max(theta)),
                "theta_x_min": float(np.min(theta_x)),
                "theta_x_mean": float(np.mean(theta_x)),
                "theta_x_median": float(np.median(theta_x)),
                "theta_x_max": float(np.max(theta_x)),
                "theta_y_min": float(np.min(theta_y)),
                "theta_y_mean": float(np.mean(theta_y)),
                "theta_y_median": float(np.median(theta_y)),
                "theta_y_max": float(np.max(theta_y)),
                "theta_frac_lt_1": float(np.mean(theta < 1.0)),
                "theta_frac_lt_0_5": float(np.mean(theta < 0.5)),
                "theta_frac_lt_0_1": float(np.mean(theta < 0.1)),
            }
        )
        for q in (1, 5, 10, 25, 75, 90, 95, 99):
            stats[f"theta_p{q:02d}"] = float(np.percentile(theta, q))

        self._add_flux_norm_stats(stats, "A", self._flux_A_x, self._flux_A_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "B", self._flux_B_x, self._flux_B_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "occupied", self._flux_A_x + self._flux_B_x, self._flux_A_y + self._flux_B_y, theta_x, theta_y)

        rho_A = self.rho_A
        rho_B = self.rho_B
        rho0 = self._rho0
        U_A = self._U_A
        U_B = self._U_B
        inv_dx = 1.0 / self.dx
        inv_dy = 1.0 / self.dy
        rhoA_r = np.roll(rho_A, -1, axis=1)
        rhoB_r = np.roll(rho_B, -1, axis=1)
        rho0_r = np.roll(rho0, -1, axis=1)
        UA_r = np.roll(U_A, -1, axis=1)
        UB_r = np.roll(U_B, -1, axis=1)
        rhoA_x = 0.5 * (rho_A + rhoA_r)
        rhoB_x = 0.5 * (rho_B + rhoB_r)
        rho0_x = 0.5 * (rho0 + rho0_r)
        det_A_x = self.params.D_A * (rho0_x * (rhoA_r - rho_A) * inv_dx - rhoA_x * (rho0_r - rho0) * inv_dx - self.params.beta * rho0_x * rhoA_x * (UA_r - U_A) * inv_dx)
        det_A_x += self.params.D_v * (rhoB_x * (rhoA_r - rho_A) * inv_dx - rhoA_x * (rhoB_r - rho_B) * inv_dx)
        det_B_x = self.params.D_B * (rho0_x * (rhoB_r - rho_B) * inv_dx - rhoB_x * (rho0_r - rho0) * inv_dx - self.params.beta * rho0_x * rhoB_x * (UB_r - U_B) * inv_dx)
        det_B_x += self.params.D_v * (rhoA_x * (rhoB_r - rho_B) * inv_dx - rhoB_x * (rhoA_r - rho_A) * inv_dx)
        det_A_x *= dt * inv_dx
        det_B_x *= dt * inv_dx

        rhoA_u = np.roll(rho_A, -1, axis=0)
        rhoB_u = np.roll(rho_B, -1, axis=0)
        rho0_u = np.roll(rho0, -1, axis=0)
        UA_u = np.roll(U_A, -1, axis=0)
        UB_u = np.roll(U_B, -1, axis=0)
        rhoA_y = 0.5 * (rho_A + rhoA_u)
        rhoB_y = 0.5 * (rho_B + rhoB_u)
        rho0_y = 0.5 * (rho0 + rho0_u)
        det_A_y = self.params.D_A * (rho0_y * (rhoA_u - rho_A) * inv_dy - rhoA_y * (rho0_u - rho0) * inv_dy - self.params.beta * rho0_y * rhoA_y * (UA_u - U_A) * inv_dy)
        det_A_y += self.params.D_v * (rhoB_y * (rhoA_u - rho_A) * inv_dy - rhoA_y * (rhoB_u - rho_B) * inv_dy)
        det_B_y = self.params.D_B * (rho0_y * (rhoB_u - rho_B) * inv_dy - rhoB_y * (rho0_u - rho0) * inv_dy - self.params.beta * rho0_y * rhoB_y * (UB_u - U_B) * inv_dy)
        det_B_y += self.params.D_v * (rhoA_y * (rhoB_u - rho_B) * inv_dy - rhoB_y * (rhoA_u - rho_A) * inv_dy)
        det_A_y *= dt * inv_dy
        det_B_y *= dt * inv_dy

        self._add_flux_norm_stats(stats, "deterministic_A", det_A_x, det_A_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "deterministic_B", det_B_x, det_B_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "deterministic_occupied", det_A_x + det_B_x, det_A_y + det_B_y, theta_x, theta_y)
        noise_A_x = self._flux_A_x - det_A_x if add_noise else np.zeros_like(det_A_x)
        noise_A_y = self._flux_A_y - det_A_y if add_noise else np.zeros_like(det_A_y)
        noise_B_x = self._flux_B_x - det_B_x if add_noise else np.zeros_like(det_B_x)
        noise_B_y = self._flux_B_y - det_B_y if add_noise else np.zeros_like(det_B_y)
        self._add_flux_norm_stats(stats, "schelling_noise_A", noise_A_x, noise_A_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "schelling_noise_B", noise_B_x, noise_B_y, theta_x, theta_y)
        self._add_flux_norm_stats(stats, "schelling_noise_occupied", noise_A_x + noise_B_x, noise_A_y + noise_B_y, theta_x, theta_y)

        limiting_activity = ((1.0 - theta_x) + np.roll(1.0 - theta_x, 1, axis=1) + (1.0 - theta_y) + np.roll(1.0 - theta_y, 1, axis=0)) * 0.25
        occ = rho_A + rho_B
        grad_x = (np.roll(occ, -1, axis=1) - np.roll(occ, 1, axis=1)) * (0.5 * inv_dx)
        grad_y = (np.roll(occ, -1, axis=0) - np.roll(occ, 1, axis=0)) * (0.5 * inv_dy)
        grad_rho = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        stats["limiting_corr_rho_A"] = self._safe_corrcoef(limiting_activity, rho_A)
        stats["limiting_corr_rho_B"] = self._safe_corrcoef(limiting_activity, rho_B)
        stats["limiting_corr_rho_0"] = self._safe_corrcoef(limiting_activity, rho0)
        stats["limiting_corr_grad_rho"] = self._safe_corrcoef(limiting_activity, grad_rho)
        return stats

    def conservative_limited_periodic_step(self, dt: float, add_noise: bool) -> None:
        """Advance one fully-active periodic step with conservative flux limiting.

        The explicit deterministic and Schelling stochastic increments are built
        as shared face increments and then rescaled by common face coefficients
        computed from cellwise A, B, and vacancy budgets.  This is deliberately
        more expensive than the clip limiter, but it keeps all face contributions
        antisymmetric and therefore preserves the finite-volume mass balances.
        Voter noise is a cell-local A<->B exchange; clipping only that exchange
        interval preserves occupied mass exactly up to roundoff.
        """
        if not self.fully_active_periodic:
            raise NotImplementedError(
                "limiter_mode='conservative' is currently implemented for fully active periodic grids only; "
                "use limiter_mode='clip' or 'none' for masked/no-flux grids."
            )
        if add_noise:
            self._fill_standard_normal(self._eta_A_x)
            self._fill_standard_normal(self._eta_B_x)
            self._fill_standard_normal(self._eta_A_y)
            self._fill_standard_normal(self._eta_B_y)
            self._fill_standard_normal(self._xi)

        limited_cells, limited_faces, voter_limited_cells = _conservative_limited_update_periodic_numba(
            self.rho_A,
            self.rho_B,
            self.dx,
            self.dy,
            self.vol,
            self.params.D_A,
            self.params.D_B,
            self.params.D_v,
            self.params.beta,
            self.params.h_noise,
            self.params.spatial_dim,
            self.params.kappa[0, 0],
            self.params.kappa[0, 1],
            self.params.kappa[1, 0],
            self.params.kappa[1, 1],
            dt,
            1 if add_noise else 0,
            self._rho0,
            self._U_A,
            self._U_B,
            self._eta_A_x,
            self._eta_B_x,
            self._eta_A_y,
            self._eta_B_y,
            self._xi,
            self._flux_A_x,
            self._flux_B_x,
            self._flux_A_y,
            self._flux_B_y,
            self._rhs_A,
            self._rhs_B,
            self._budget_A,
            self._budget_B,
            self._budget_occ,
            self._alpha_A,
            self._alpha_B,
            self._alpha_occ,
            self._rho_A_new,
            self._rho_B_new,
        )
        # Collect face-based diagnostics before committing the output state so
        # localization and deterministic/noise decomposition use the pre-step
        # densities that generated the raw face increments.
        self.last_limiter_stats = self._collect_conservative_limiter_diagnostics(
            dt,
            add_noise,
            int(limited_cells),
            int(limited_faces),
            int(voter_limited_cells),
            0,
            0.0,
        )
        gamma_repair_cells = 0
        gamma_repair_residual = 0.0
        if self.semi_implicit_stiff:
            target_A = float(np.sum(self._rho_A_new) * self.vol)
            target_B = float(np.sum(self._rho_B_new) * self.vol)
            rho_A_new, rho_B_new = self.semi_implicit_gamma_step(self._rho_A_new, self._rho_B_new, dt)
            self.rho_A[...] = rho_A_new
            self.rho_B[...] = rho_B_new
            gamma_repair_cells, gamma_repair_residual = self._conservative_global_simplex_repair(
                self.rho_A, self.rho_B, target_A, target_B
            )
        else:
            self.rho_A[...] = self._rho_A_new
            self.rho_B[...] = self._rho_B_new
        self.last_limiter_corrections = int(limited_cells + voter_limited_cells + gamma_repair_cells)
        self.last_limiter_stats["gamma_repair_cells"] = int(gamma_repair_cells)
        self.last_limiter_stats["activations"] = 1 if (limited_cells > 0 or limited_faces > 0 or voter_limited_cells > 0 or gamma_repair_cells > 0) else 0
        self.last_limiter_stats["residual_max_violation"] = max(self._max_simplex_violation(), float(gamma_repair_residual))

    def step(self, dt: float, add_noise: Optional[bool] = None) -> None:
        if add_noise is None:
            add_noise = self.stochastic

        if self.limiter_mode == "conservative":
            self.conservative_limited_periodic_step(dt, bool(add_noise))
            return

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
        if self.limiter_mode == "clip":
            limiter_corrections += self.apply_simplex_limiter(self._rho_A_star, self._rho_B_star)

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
            limiter_corrections += self.apply_simplex_limiter()
            self.last_limiter_corrections = limiter_corrections
            self.last_limiter_stats = self._empty_limiter_stats(self.limiter_mode)
            self.last_limiter_stats.update(
                {
                    "limited_cells": int(limiter_corrections),
                    "activations": 1 if limiter_corrections > 0 else 0,
                    "residual_max_violation": self._max_simplex_violation(),
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
        limiter_limited_faces = []
        limiter_voter_limited_cells = []
        limiter_gamma_repair_cells = []
        limiter_activations = []
        limiter_residual_max_violation = []
        limiter_diagnostic_keys = [
            "theta_min",
            "theta_mean",
            "theta_median",
            "theta_frac_lt_1",
            "theta_frac_lt_0_5",
            "theta_frac_lt_0_1",
            "flux_A_removed_fraction",
            "flux_B_removed_fraction",
            "flux_occupied_removed_fraction",
            "flux_deterministic_occupied_removed_fraction",
            "flux_schelling_noise_occupied_removed_fraction",
            "limiting_corr_rho_A",
            "limiting_corr_rho_B",
            "limiting_corr_rho_0",
            "limiting_corr_grad_rho",
        ]
        limiter_diagnostics = {key: [] for key in limiter_diagnostic_keys}
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
                limiter_limited_faces.append(int(self.last_limiter_stats.get("limited_faces", 0)))
                limiter_voter_limited_cells.append(int(self.last_limiter_stats.get("voter_limited_cells", 0)))
                limiter_gamma_repair_cells.append(int(self.last_limiter_stats.get("gamma_repair_cells", 0)))
                limiter_activations.append(int(self.last_limiter_stats.get("activations", 0)))
                limiter_residual_max_violation.append(float(self.last_limiter_stats.get("residual_max_violation", 0.0)))
                for key in limiter_diagnostic_keys:
                    limiter_diagnostics[key].append(float(self.last_limiter_stats.get(key, 0.0)))
                # fhd diagnostics expect species-first fields with shape (2, nx, ny).
                phi = np.array([self.rho_A.T, self.rho_B.T])
                dissimilarity_index = fhd.dissimilarity(phi)
                mean_kl_divergence = fhd.mean_relative_entropy(phi)
                print(
                    f"step {n}/{nsteps}:    "
                    f"<m_A> = {masses_A[-1]/masses_tot[-1]:.2f},   "
                    f"<m_B> = {masses_B[-1]/masses_tot[-1]:.2f},    "
                    f"<m_occ> = {masses_occ[-1]/masses_tot[-1]:.2f},    "
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
            "limiter_limited_faces": np.array(limiter_limited_faces),
            "limiter_voter_limited_cells": np.array(limiter_voter_limited_cells),
            "limiter_gamma_repair_cells": np.array(limiter_gamma_repair_cells),
            "limiter_activations": np.array(limiter_activations),
            "limiter_residual_max_violation": np.array(limiter_residual_max_violation),
        }
        for key, values in limiter_diagnostics.items():
            out[f"limiter_{key}"] = np.array(values)
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
