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

from .operations import *

class fhd_2d:
    '''Defines the 2-D fluctuating hydrodynamics class for simulating the sociohydrodynamic equations including noise and reactions:

    '''
    def __init__(self, L, N, bc = "periodic", fft = False, schelling_flux="collocated"):
        '''
        Initializes instance of the fhd class object

        Args:
            L:   tuple (Lx, Ly): spatial lengths of the domain, coordinates will be defined as running from -L/2 to L/2
            N:   tuple (Nx, Ny): number of discretization steps per coordinate
            bc:  boundary conditions, choose "periodic" or "Neumann"
            fft: Bool: when True derivatives are computed using FFT (only compatible with periodic bc's)
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

        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)*2*np.pi
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)*2*np.pi
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        self.phi_floor = 1e-14
        self.nspecies = 2

        # Matrices Dx and Dy for 8-th order finite differences, needed for divergence function below
        self.Dx = makeD(self.Nx, self.dx, self.bc)
        self.Dy = makeD(self.Ny, self.dy, self.bc)
        if not fft:
            self.D2x = makeD2(self.Nx, self.dx, self.bc)
            self.D3x = makeD3(self.Nx, self.dx, self.bc)
            
            self.D2y = makeD2(self.Ny, self.dy, self.bc)
            self.D3y = makeD3(self.Ny, self.dy, self.bc)


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

        return self._work
        
    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def scale_down_pointwise(self, phi):
        ''' Scales down phi[a] on sites where sum_a phi[a] > 1
        '''
        sumphi = np.sum(phi, axis=0)
        divisor = np.maximum(sumphi+self.phi_floor,1)
        # scale down phi only at point where sum exceeds one
        phi = phi/divisor
        return phi
    
    def project_density(self, rho):
        # Always clip in-place; avoids separate np.any scans.
        np.maximum(rho, self.phi_floor, out=rho)
        np.minimum(rho, 1.0 - self.phi_floor, out=rho)

        sumrho = rho[0] + rho[1]
        mask = sumrho > 1.0

        if np.any(mask):
            rho[0, mask] /= sumrho[mask] + self.phi_floor
            rho[1, mask] /= sumrho[mask] + self.phi_floor

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
        np.maximum(amp_x, self.phi_floor**2, out=amp_x)
        np.maximum(amp_y, self.phi_floor**2, out=amp_y)

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
            np.maximum(rho_ab, self.phi_floor**2, out=rho_ab)

            # demo_noise = sqrt(4 D_v rho_ab / dt) * xi2
            np.multiply(rho_ab, 4.0 * D_v / dt, out=demo_noise)
            np.sqrt(demo_noise, out=demo_noise)
            demo_noise *= xi2

            # Apply your old clipping/capping logic, but in-place-ish.
            noise_v = param.get("noise_v", 1.0)

            # Reuse rho_ab as temporary lower/upper clipped noise.
            # rho_ab = max(-phi_A/dt, noise_v * demo_noise)
            np.multiply(demo_noise, noise_v, out=rho_ab)
            np.maximum(rho_ab, -phi[0] / dt, out=rho_ab)

            # rho_ab = min(rho_ab, phi_B/dt)
            np.minimum(rho_ab, phi[1] / dt, out=rho_ab)

            divJ[0] += rho_ab
            divJ[1] -= rho_ab
        
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
        
    def step(self, rhs, phi, param, dt, toggle_noise, scheme, work = None):
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

        return self.project_density(rho_next)

    def scale_down(self, phi):
        ''' Scales down phi such that  sum_a phi[a] < 1 for all x 
        '''
        sumphi = np.sum(phi, axis=0)
        if np.any(sumphi >= 1 - self.phi_floor):
            # scale down slightly
            phi *= (1.0 / (sumphi.max() + self.phi_floor))
        return phi

    def run(self, phi, param, nsteps, dt, toggle_noise, 
            no_frames = 100,
            scheme = "FE", 
            model = "Vitelli",
            verbatum = True):
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
        phi = self.project_density(phi)
        phi = np.maximum(phi, self.phi_floor)
        
        phi_current = phi.copy()
        phi_run = np.zeros((self.nspecies, no_frames+1)+self.N)
        phi_run[:,0,:,:] = phi_current
        
        for n in range(1, nsteps + 1):     
            # print("step", n)
            if model == "Vitelli":
                phi_current = self.step(self.rhs_Vitelli, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling":
                phi_current = self.step(self.rhs_Schelling, phi_current, param, dt, toggle_noise, scheme)
            elif model == "Schelling+Voter":
                work = self._ensure_work(phi_current.dtype)
                phi_current = self.step(self.rhs_SchellingwithVoter, phi_current, param, dt, toggle_noise, scheme, work)
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
                        'noise_v': float: strength of voter model noise (default is one)
            verbatum: If True print and plot stuff

        Returns:
            phi_run: numpy array of shape (nspecies, frames+1, N) with the simulation timeseries
                    
        '''
        plot_every = save_every
        phi = self.project_density(phi)
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
