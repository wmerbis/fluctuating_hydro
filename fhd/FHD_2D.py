"""
1D implementation of sociohydrodynamic models based on Schelling's model.

Model "Vitelli" incorporates a noisy version of the model presented in Seara et.al 2025. With local density phi_a 
of agents of type a evolving as:

∂t phi_a = ∂x [D phi_0 ∂x phi_a - D phi_a ∂x + phi_a phi_0 ∂x pi_a +  sqrt(2D dx^d phi_a phi_0) Z] + R(phi)

with: 
phi_0 = 1 - sum_b phi_b (density of vacant sites) 
Z Gaussian white noise
R(phi) implements a voter model with mean-field equation ∂t phi_a = phi*(b*pho_0 - d) with b and d birth and death rates respectively

Model "Schelling" implements Schelling type rules where agents only diffuse if the fraction of like agents is below a threshold theta.
In that case the dynamical equations become:

∂t phi_a =   w0 (phi_0 ∂^2 phi_a - phi_a ∂^2 phi_0)  + phi_a * phi_0 * ∂^2 w0 + 2 phi_0 ∂ w0 . ∂ phi_a ) +  ∂[sqrt(2 w0 dx^d phi_a phi_0) Z] + R(phi)

with:
w0 = D/ (1+ e^(-\beta \pi)) density dependent diffusion term
pi = \sum_b ((theta - 1) delta^{ab} + theta sigma_x^{ab} ) (phi_b + \Gamma ∂^2 phi_b) utility threshold function with Gaussian smeared neighborhood (\sigma^2/2 = \Gamma)
Z Gaussian white noise
R(phi) implements a voter model with mean-field equation ∂t phi_a = phi*(b*pho_0 - d) with b and d birth and death rates respectively

Features:
- Two class objects: fhd for 1D, fhd_2d for 2D
- Positivity floor for densities
- multiplicative conservative noise ~ sqrt(phi_a phi_0), turn on by passing non-zero "toggle_noise"
- numerical differentiation by fft, when passing "fft = False" when initializing the class objects derivatives are computed using finite difference 

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
    '''Defines the 1-D fluctuating hydrodynamics class for simulating the sociohydrodynamic equations including noise and reactions:

    '''
    def __init__(self, L, N, bc = "periodic", fft = False):
        '''
        Initializes instance of the fhd class object

        Args:
            L:  tuple (Lx, Ly): spatial lengths of the domain, coordinates will be defined as running from -L/2 to L/2
            N:  tuple (Nx, Ny): number of discretization steps per coordinate
            bc: boundary conditions, choose among "periodic", "Neumann" or "Dirichlet"
                (note: latter two are to be implemented
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
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'Neumann' or 'Dirichlet' ")
        
        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)*2*np.pi
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)*2*np.pi
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        self.phi_floor = 1e-14
        self.nspecies = 2

        # Matrixces Dx and Dy for 8-th order finite differences, needed for divergence function below
        self.Dx = makeD(self.Nx, self.dx, self.bc)
        self.Dy = makeD(self.Ny, self.dy, self.bc)
        if not fft:
            self.D2x = makeD2(self.Nx, self.dx, self.bc)
            self.D3x = makeD3(self.Nx, self.dx, self.bc)
            
            self.D2y = makeD2(self.Ny, self.dy, self.bc)
            self.D3y = makeD3(self.Ny, self.dy, self.bc)
        
    def scale_down_pointwise(self, phi):
        ''' Scales down phi[a] on sites where sum_a phi[a] < 1
        '''
        sumphi = np.sum(phi, axis=0)
        divisor = np.maximum(sumphi+self.phi_floor,1)
        # scale down phi only at point where sum exceeds one
        phi = phi/divisor
        return phi

    def grad(self, u):
        if self.fft:
            u_hat = np.fft.fft2(u)
        
            grad_x_hat = 1j * self.kx * u_hat
            grad_y_hat = 1j * self.ky * u_hat
        
            grad = np.zeros((2,)+u.shape)
            grad[0] = np.fft.ifft2(grad_x_hat).real
            grad[1] = np.fft.ifft2(grad_y_hat).real
        else:
            ush = u.shape
            grad = np.zeros((2,)+ush)
            grad[0] = np.array([self.Dx.dot(u[i]) for i in range(ush[0])])
            grad[1] = np.array([self.Dy.dot(u[i].T).T for i in range(ush[0])])
        return grad

        
    def lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft2(u)
            
            lapl_hat = -(self.kx**2 + self.ky**2) * u_hat
            lapl = np.fft.ifft2(lapl_hat).real
        else:
            ush = u.shape
            lapl = np.array([self.D2x.dot(u[i]) for i in range(ush[0])])
            lapl += np.array([self.D2y.dot(u[i].T).T for i in range(ush[0])])          
        return lapl

    def grad_lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft2(u)
            
            gradlap_x_hat = - 1j * self.kx * (self.kx**2 + self.ky**2) * u_hat
            gradlap_y_hat = - 1j * self.ky * (self.kx**2 + self.ky**2) * u_hat
        
            gradlap = np.zeros((2,)+u.shape)
            gradlap[0] = np.fft.ifft2(gradlap_x_hat).real
            gradlap[1] = np.fft.ifft2(gradlap_y_hat).real
        else:
            ush = u.shape
            gradlap = np.zeros((2,)+ush)
            gradlap[0] =  [self.D3x.dot(u[i]) for i in range(ush[0])] 
            gradlap[0] += [self.Dx.dot(self.D2y.dot(u[i].T).T) for i in range(ush[0])]
            gradlap[1] =  [self.D3y.dot(u[i].T).T for i in range(ush[0])] 
            gradlap[1] += [self.Dy.dot(self.D2x.dot(u[i]).T).T for i in range(ush[0])]
        return gradlap

    def grad_utility(self, phi, param):
        """Compute ∇ U = (∇πa + Γa ∇^3ρa) with πa = sum_b kappa_ab * rho_b for each species a."""
        pi = np.einsum("ab, bij -> aij", param['kappa'],phi)
        Gammaterm = np.einsum("ab, xbij -> xaij", param['Gamma'], self.grad_lapl(phi))
        if 'nu' in param:
            phisqr_ab = np.einsum("aij, bij -> abij", phi, phi)
            nuterm = np.einsum("abc, bcij-> aij", param['nu'], phisqr_ab, axes = ([1,2],[0,1]))
            return self.grad(pi) + self.grad(nuterm) + Gammaterm
        else:
            return self.grad(pi) + Gammaterm

    def div2d(self, vec):
        '''
        Divergence of a spatial vector: NOTE this uses finite differences instead of FFT, for some reason FFT gives bad results 
        when computing divergence of derivatives! To be understood properly...
        '''
        # vec_hat = np.fft.fft2(vec)
        
        # div = np.fft.ifft2(- 1j * self.kx * np.fft.fft2(vec[0])).real
        # div += np.fft.ifft2(- 1j * self.ky * np.fft.fft2(vec[1])).real
        div = np.array([self.Dx.dot(vec[0,i]) for i in range(self.nspecies)])
        div += np.array([self.Dy.dot(vec[1,i].T).T for i in range(self.nspecies)])
        return div
    
    def rhs_Vitelli(self, phi, param, dt, toggle_noise):
        """Compute RHS of the equation"""

        phi0   = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,)+self.N)
        dUdx = self.grad_utility(phi, param)
        
    
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        rhorho0dUdx = phi*phi0*dUdx
        div_dUdx = self.div2d(rhorho0dUdx) 
        D = param['D']
        divJ = np.einsum("a, aij->aij", D, phi0*self.lapl(phi) - phi*self.lapl(phi0)) - div_dUdx 
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            xi = np.random.normal(0, 1, size= (2,)+phi.shape)
            if self.bc == 'Neumann':
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,:,0,:] = 0
                xi[:,:,-1,:] = 0
                xi[:,:,:,0] = 0
                xi[:,:,:,-1] = 0
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_term   = np.einsum("a, aij-> aij", param['D'], rho_face)*2*self.dx*self.dy/dt
            noise_flux   = np.einsum("aij, laij -> laij", np.sqrt(noise_term), xi) 
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ

    def rhs_SchellingwithVoter(self, phi, param, dt, toggle_noise):
        """Compute RHS of the equation"""

        phi0   = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,)+self.N)
        dUdx = self.grad_utility(phi, param)
        
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        rhorho0dUdx = phi*phi0*dUdx
        lapphi = self.lapl(phi)
        div_dUdx = self.div2d(rhorho0dUdx) 
        D = param['D']
        divJ = np.einsum("a,aij -> aij", D, phi0*lapphi - phi*self.lapl(phi0) - param['beta']*div_dUdx)
        voter_current = param['D_v']*(phi[1]*lapphi[0] - phi[0]*lapphi[1])
        divJ[0] += voter_current
        divJ[1] += -voter_current
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            # Add conservative noise for Voter
            xi = np.random.normal(0, 1, size= (2,)+phi.shape)
            if self.bc == 'Neumann':
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,:,0,:] = 0
                xi[:,:,-1,:] = 0
                xi[:,:,:,0] = 0
                xi[:,:,:,-1] = 0
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_term   = np.einsum("a, aij-> aij", param['D'], rho_face)*2*self.dx*self.dy/dt
            noise_flux   = np.einsum("aij, laij -> laij", np.sqrt(noise_term), xi)             
            dnoise_dx    = self.div2d(noise_flux) 
            divJ += toggle_noise*dnoise_dx

            # Add demographic noise for Voter model
            xi2 = np.random.normal(0, 1, size = self.N)
            # if self.bc == "Neumann":
            #     xi2[0,:] = 0
            #     xi2[-1,:] = 0
            #     xi2[:,0] = 0
            #     xi2[:,-1] = 0
            rho_face     = np.maximum(phi[0]*phi[1], self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            demographic_noise = np.einsum("ij,ij->ij", np.sqrt(4*param['D_v']*rho_face/dt), xi2)
            
            noise_ceiling = np.maximum(-phi[0]/dt,param['noise_v']*demographic_noise)
            noise_floor = np.minimum(noise_ceiling, phi[1]/dt)
            divJ[0] += noise_floor
            divJ[1] -= noise_floor
        
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
        divJ = w0*phi0*self.lapl(phi) - w0*phi*self.lapl(phi0)  + phi*phi0*laplw0 + 2*phi0*(gradw0[0]*gradphi[0] + gradw0[1]*gradphi[1])
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            xi = np.random.normal(0, 1, size= (2,) + phi.shape)
            if self.bc == 'Neumann':
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,:,0,:] = 0
                xi[:,:,-1,:] = 0
                xi[:,:,:,0] = 0
                xi[:,:,:,-1] = 0
            rho_face     = np.maximum(w0*phi*phi0, 1e-14) 
            noise_flux   = np.einsum("aij, laij -> laij", np.sqrt(2*self.dx*self.dy*rho_face/dt), xi)
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ
        
    def step(self, rhs, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred # + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            rho_corr = phi + 0.5*dt*(dphidt + rhs(rho_pred, param,  dt, toggle_noise))
            rho_next = rho_corr # + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "RK4":
            k1 = dt * dphidt
            k2 = dt * rhs(phi + 1/2* k1, param, dt, toggle_noise)
            k3 = dt * rhs(phi + 1/2* k2, param, dt, toggle_noise)
            k4 = dt * rhs(phi + k3, param, dt, toggle_noise)
            rho_next = phi + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            # rho_next +=  dt * phi*(param['b']*(1-phi_tot) - param['d'])
        
        if np.any(rho_next <0):
            # print("fluctuations made phi[a] negative")
            rho_next = np.maximum(rho_next,self.phi_floor)
        
        if np.any(rho_next>1):
            # print("fluctuations made phi[a] larger than 1")
            rho_next = np.minimum(rho_next,1-self.phi_floor)
        
        if np.any(rho_next.sum(axis=0)>1):
            rho_next = self.scale_down_pointwise(rho_next)
            # raise ValueError("Fluctuations made phi0 negative")      
        
        return rho_next

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
        phi = self.scale_down(phi)
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
                phi_current = self.step(self.rhs_SchellingwithVoter, phi_current, param, dt, toggle_noise, scheme)
            else:
                raise ValueError(f"Model {model} is unknown, please choose 'Vitelli', 'Schelling' or 'Schelling+Voter'") 
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {dissimilarity(phi_current):.6f}, H_index = {mean_relative_entropy(phi_current):.6f}")
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
            plt.ylabel("t")
            cbar = plt.colorbar(im, fraction=0.046)
            cbar.set_label(r"$\phi_a - \phi_b$",size=14)
            plt.show()
            
        return phi_run
