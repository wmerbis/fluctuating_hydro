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
Expanded to three types by Rinske Oskamp
""" 
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.fft import dct, idct
import scipy as sp

from fhd.operations import *

class fhd_3species:
    '''Defines the 1-D fluctuating hydrodynamics class for simulating sociohydrodynamic models including noise and reactions

    '''
    def __init__(self, L, N, bc = "periodic", fft = False):
        '''
        Initializes instance of the fhd class object

        Args:
            L:   length of the spatial domain, x coordinate will be defined as running from -L/2 to L/2
            N:   number of discretization steps
            bc:  boundary conditions, choose among "periodic", "Neumann" or "Dirichlet"
                 (note: latter two are to be implemented)
            fft: Boolean, when True, derivatives are computed using fast Fourier transforms.
                 when False, 6th order finite difference methods are used
        '''
        
        self.L = L
        self.dx = L / N
        self.bc = bc
        self.fft = fft
        
        if bc == "periodic":
            self.N = N
            self.x = np.arange(-L/2, L/2, self.dx)
        elif bc == "Neumann":
            self.N = N + 1
            self.x = np.linspace(-L/2, L/2, self.N)
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'Neumann' or 'Dirichlet' ")

        if fft and bc == 'periodic':
            self.k = 2*np.pi*np.fft.fftfreq(self.N, d=self.dx)
        elif fft and bc == 'Neumann':
            # self.k = np.pi * np.arrange(self.N) /self.L 
            raise ValueError("Neumann boundary conditions not implemented for fft derivatives")
        else:
            self.Dx = makeD(self.N, self.dx, self.bc)
            self.D2x = makeD2(self.N, self.dx, self.bc)
            self.D3x = makeD3(self.N, self.dx, self.bc)
            # self.D4x = makeD4(self.N, self.dx, self.bc)
            
        self.phi_floor = 1e-14
        self.nspecies = 3
        
    def scale_down_pointwise(self, phi):
        ''' Scales down phi[a] on sites where sum_a phi[a] < 1
        '''
        sumphi = np.sum(phi, axis=0)
        divisor = np.maximum(sumphi+self.phi_floor,1)
        # scale down phi only at point where sum exceeds one
        phi = phi/divisor
        return phi

    def grad(self, u):
        if self.fft and self.bc == 'periodic':
            u_hat = np.fft.fft(u)
            grad = np.fft.ifft(1j*self.k*u_hat).real
        elif self.fft and self.bc == 'Neumann':
            u_hat = dct(u, type=2, norm='ortho') #Neumann bc's use cosine transforms
            # factor = (self.k) 
            # factor[0] = 0  # The zero frequency component should remain zero for Neumann BCs
            grad = idct(self.k * u_hat, type=2, norm='ortho').real
        else:
            grad = self.Dx.dot(u.T).T
        return grad
        
    def lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft(u)
            lapl = np.fft.ifft(- self.k**2*u_hat).real
        else:
            lapl = self.D2x.dot(u.T).T
        return lapl

    def grad_lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft(u)
            d3 = np.fft.ifft(- 1j*self.k**3*u_hat).real
        else:
            d3 = self.D3x.dot(u.T).T
        return d3

    def grad_utility(self, phi, param):
        """Compute ∇ U = (∇πa + Γa ∇^3ρa) with πa = sum_b kappa_ab * rho_b for each species a."""
        pi = np.einsum("ab, b... -> a...", param['kappa'], phi)
        Gammaterm = np.einsum("ab, b... -> a...", param['Gamma'], self.grad_lapl(phi))
        
        if 'nu' in param:
            phisqr_ab = np.einsum("b..., c... -> bc...", phi, phi)
            nuterm = np.einsum("abc, bc... -> a...", param['nu'], phisqr_ab)
            return self.grad(pi) + self.grad(nuterm) + Gammaterm
        else:
            return self.grad(pi) + Gammaterm
        
    def rhs_Vitelli(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""
        phi0 = 1- phi.sum(axis=0)
        dUdx = self.grad_utility(phi, param)
    
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        D = param['D']

        # # Diffusive terms
        # term1 = phi0 * self.lapl(phi)           # (a, i, j)
        # term2 = phi * self.lapl(phi0)           # (a, i, j)

        # # Drift term: ∇·(φ φ0 ∇U)
        # drift_flux = phi * phi0 * dUdx          # (x, a, i, j)
        # term3 = self.div(drift_flux)            # (a, i, j)

        # divJ = np.einsum(
        #     "a, a... -> a...",
        #     D,
        #     term1 - term2 - param["beta"] * term3
        # )

        divJ = np.einsum("a, ai -> ai", D, phi0*self.lapl(phi) - phi*self.lapl(phi0) - param['beta']*self.grad(phi*phi0*dUdx) )
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            if 'h' in param:
                h = param['h']
            else:
                h = self.dx
            xi = np.random.normal(0, 1, size= phi.shape)
            if self.bc == "Neumann":
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,0] = 0
                xi[:,-1] = 0
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_term   = np.einsum("a, a...-> a...", D, rho_face)*2*h/dt
            # noise_flux   = np.einsum("ai, ai -> ai", np.sqrt(noise_term), xi) 
            noise_flux = np.sqrt(noise_term) * xi
            dnoise_dx    = self.grad(noise_flux)
        
            divJ += dnoise_dx
        
        return divJ

    def w0(self, phi, param):
        D = param["D"]
        beta = param["beta"]
        theta = param["theta"]
        sigma = param["sigma"]
        kappa = np.array([[theta-1,theta],[theta,theta-1]])
        Gamma = kappa*sigma**2/2
        pi = np.dot(kappa, phi) + np.dot(Gamma, self.lapl(phi))
        w0 = np.einsum("a, ai -> ai", D, 1/(1+np.exp(-beta*pi)))
        grad_pi = np.dot(kappa, self.grad(phi))
        grad_pi += np.dot(Gamma, self.grad_lapl(phi))
        gradw0 = np.einsum("ai, ai -> ai",  np.einsum("a, ai->ai", D, beta/(2+2*np.cosh(beta*pi))) , grad_pi)
        laplw0 = self.grad(gradw0)
        return w0, gradw0, laplw0


    def rhs_Schelling(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""
        phi0 = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,self.N))
        w0, gradw0, laplw0 = self.w0(phi, param)
        
        """Compute grad J =   w0 (rho_0 ∂^2 rho -  rho ∂^2 rho_0)  + rho * rho_0 * ∂^2 w0 + 2 rho0 ∂ w0 . ∂ rho ) """
        divJ = w0*phi0*self.lapl(phi) - w0*phi*self.lapl(phi0)  + phi*phi0*laplw0 + 2*phi0*gradw0*self.grad(phi)
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            if 'h' in param:
                h = param['h']
            else:
                h = self.dx
            xi = np.random.normal(0, 1, size= phi.shape)
            if self.bc == "Neumann":
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,0] = 0
                xi[:,-1] = 0
            rho_face     = np.maximum(w0*phi*phi0, self.phi_floor**2) 
            noise_flux   = np.sqrt(2*h*rho_face/dt)*xi # Double check noise flux dx and dt dependence!!
            dnoise_dx    = self.grad(noise_flux)
        
            divJ += dnoise_dx
        
        return divJ
    
    def rhs_SchellingwithVoter(self, phi, param, dt, toggle_noise):
        """Compute RHS of the equation"""

        phi0 = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,self.N))
        dUdx = self.grad_utility(phi, param)
        
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        rhorho0dUdx = phi*phi0*dUdx
        lapphi = self.lapl(phi)
        div_dUdx = self.grad(rhorho0dUdx) 
        D = param['D']
        divJ = np.einsum("a,ai -> ai", D, phi0*lapphi - phi*self.lapl(phi0) - param['beta']*div_dUdx)
        voter_current = param['D_v']*(phi[1]*lapphi[0] - phi[0]*lapphi[1])
        divJ[0] += voter_current
        divJ[1] += -voter_current
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            if 'h' in param:
                h = param['h']
            else:
                h = self.dx
            # Add conservative noise for Voter
            xi = np.random.normal(0, 1, size= phi.shape)
            if self.bc == "Neumann":
                # Set noise to zero on the boundary for Neumann bc's
                xi[:,0] = 0
                xi[:,-1] = 0
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_term   = np.einsum("a, ai-> ai", D, rho_face)*2*h/dt
            noise_flux   = np.einsum("ai, ai -> ai", np.sqrt(noise_term), xi)             
            dnoise_dx    = self.grad(noise_flux) 
            divJ += dnoise_dx

            # Add demographic noise for Voter model
            xi2 = np.random.normal(0, 1, size = self.N)
            # if self.bc == "Neumann":
            #     xi2[0,:] = 0
            #     xi2[-1,:] = 0
            #     xi2[:,0] = 0
            #     xi2[:,-1] = 0
            rho_face     = np.maximum(phi[0]*phi[1], self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            demographic_noise = np.einsum("i,i->i", np.sqrt(4*param['D_v']*rho_face/dt/h), xi2)
            
            noise_ceiling = np.maximum(-phi[0]/dt,param['noise_v']*demographic_noise)
            noise_floor = np.minimum(noise_ceiling, phi[1]/dt)
            divJ[0] += noise_floor
            divJ[1] -= noise_floor
        
        return divJ
    
    def step(self, rhs, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred # + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            dphidt_pred = rhs(rho_pred, param,  dt, toggle_noise)
            rho_corr = phi + 0.5*dt*(dphidt + dphidt_pred)
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
                    Vitelli"  with expected parameters in dictionary param:
                        'D':     numpy array of shape (nspecies): diffusion constants for each species
                        'beta':  Inverse temperature, large beta means scricter enforcement of utility gradient
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
                        'beta':  Inverse temperature, large beta means scricter enforcement of utility gradient
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
        phi_run = np.zeros((self.nspecies, no_frames+1, self.N))
        phi_run[:,0,:] = phi_current
        
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
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {dissimilarity(phi_current):.6f}")
                phi_run[:,n//plot_every,:] = phi_current

        if verbatum:
            phi_diff = phi_run[0,:,:] - phi_run[1,:,:]
            im =plt.imshow(phi_diff, cmap = 'RdBu', aspect='auto', origin='lower', extent=[-self.L/2,self.L/2,0,dt*nsteps], vmin=-1, vmax=1)
            if model == "Schelling":
                theta = param['theta']
                D = param['D']
                sigma = param['sigma']
                beta = param['beta']
                title = fr"$D = [{D[0]:.1f}, {D[1]:.1f}],\, \theta = {theta:.2f} \,, \sigma = {sigma:.1f}\,, \beta = {beta:.1f}$"
            elif model == "Schelling+Voter":
                kappa = param['kappa']
                D = param['D']
                Gamma = param['Gamma']
                D_v = param['D_v']
                title = fr"$D = [{D[0]:.1f}, {D[1]:.1f}],\, \kappa = [[{kappa[0,0]:.1f}, {kappa[0,1]:.1f}], [{kappa[1,0]:.1f} , {kappa[1,1]:.1f}]] \,, \Gamma = [[{Gamma[0,0]:.1f}, {Gamma[0,1]:.1f}], [{Gamma[1,0]:.1f} , {Gamma[1,1]:.1f}]], \, D_v = {D_v} $"            
            elif model == "Vitelli":
                kappa = param['kappa']
                D = param['D']
                Gamma = param['Gamma']
                title = fr"$D = [{D[0]:.1f}, {D[1]:.1f}],\, \kappa = [[{kappa[0,0]:.1f}, {kappa[0,1]:.1f}], [{kappa[1,0]:.1f} , {kappa[1,1]:.1f}]] \,, \Gamma = [[{Gamma[0,0]:.1f}, {Gamma[0,1]:.1f}], [{Gamma[1,0]:.1f} , {Gamma[1,1]:.1f}]]$"
            
            #plot individual species
            fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)
            extent = [-self.L/2, self.L/2, 0, dt*nsteps]

            im_0 =axes[0].imshow(phi_run[0,:,:], cmap = 'Reds', aspect='auto', origin='lower', extent=extent, vmin=0, vmax=1)
            # plt.title(f"Species a: {title}")
            # plt.xlabel("x")
            # plt.ylabel("t")
            # cbar = plt.colorbar(im_a, fraction=0.046)
            # cbar.set_label(r"$\phi_a - \phi_b$",size=18)
            # plt.show()

            im_1 =axes[1].imshow(phi_run[1,:,:], cmap = 'Blues', aspect='auto', origin='lower', extent=extent, vmin=0, vmax=1)
            # plt.title(f"Species b: {title}")
            # plt.xlabel("x")
            # plt.ylabel("t")
            # cbar = plt.colorbar(im_b, fraction=0.046)
            # cbar.set_label(r"$\phi_a - \phi_b$",size=18)
            # plt.show()

            im_2 =axes[2].imshow(phi_run[2,:,:], cmap = 'Greens', aspect='auto', origin='lower', extent=extent, vmin=0, vmax=1)
            # plt.title(f"Species c: {title}")
            # plt.xlabel("x")
            # plt.ylabel("t")
            # cbar = plt.colorbar(im_c, fraction=0.046)
            # cbar.set_label(r"$\phi_a - \phi_b$",size=18)
            # plt.show()

            

            axes[0].set_title(f"Species a: {title}")
            axes[0].set_ylabel("t")
            fig.colorbar(im_0, ax=axes[0], fraction=0.046)

            axes[1].set_title(f"Species b: {title}")
            axes[1].set_ylabel("t") 
            fig.colorbar(im_1, ax=axes[1], fraction=0.046)

            axes[2].set_title(f"Species c: {title}")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("t")
            fig.colorbar(im_2, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            plt.show()

            # Plot combined 
            plt.figure()

            im_a = plt.imshow(
                phi_run[0,:,:],
                cmap='Reds',
                alpha=0.8,
                aspect='auto',
                origin='lower',
                extent=[-self.L/2, self.L/2, 0, dt*nsteps],
                vmin=0,
                vmax=1
            )

            im_b = plt.imshow(
                phi_run[1,:,:],
                cmap='Blues',
                alpha=0.6,
                aspect='auto',
                origin='lower',
                extent=[-self.L/2, self.L/2, 0, dt*nsteps],
                vmin=0,
                vmax=1
            )

            im_c = plt.imshow(
                phi_run[2,:,:],
                cmap='Greens',
                alpha=0.6,
                aspect='auto',
                origin='lower',
                extent=[-self.L/2, self.L/2, 0, dt*nsteps],
                vmin=0,
                vmax=1
            )

            plt.title(f"Species a + b + c: {title}")
            plt.xlabel("x")
            plt.ylabel("t")

            plt.show()

            
            plt.plot(self.x, phi_current[0], lw=2, color= 'C0', label = r'$\phi_a$')
            plt.plot(self.x, phi_current[1], lw=2, color= 'C1', label = r'$\phi_b$')
            plt.plot(self.x, phi_current[2], lw=2, color= 'C2', label = r'$\phi_c$')
            plt.title("Final state")
            plt.ylim(0, 1.2)
            plt.xlabel('x')
            plt.ylabel('phi(x,t)')
            plt.legend()
            plt.show()
            
        return phi_run
