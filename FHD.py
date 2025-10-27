"""
1D and 2D implementation of sociohydrodynamic models based on Schelling's model.

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


class fhd:
    '''Defines the 1-D fluctuating hydrodynamics class for simulating sociohydrodynamic models including noise and reactions

    '''
    def __init__(self, L, N, bc = "periodic", fft = True):
        '''
        Initializes instance of the fhd class object

        Args:
            L:   length of the spatial domain, x coordinate will be defined as running from -L/2 to L/2
            N:   number of discretization steps
            bc:  boundary conditions, choose among "periodic", "von Neumann" or "Dirichlet"
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
        elif bc == "von Neumann":
            raise ValueError("von Neumann boundary conditions not yet implemented")
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'von Neumann' or 'Dirichlet' ")

        if fft:
            self.k = np.fft.fftfreq(N, d=self.dx)*2*np.pi
        else:
            self.Dx = makeD(self.N, self.dx)
            self.D2x = makeD2(self.N, self.dx)
            self.D3x = makeD3(self.N, self.dx)
            self.D4x = makeD4(self.N, self.dx)
            
        self.phi_floor = 1e-14
        self.nspecies = 2
        
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
            u_hat = np.fft.fft(u)
            grad = np.fft.ifft(1j*self.k*u_hat).real
        else:
            grad = np.dot(self.Dx, u.T).T
        return grad
        
    def lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft(u)
            lapl = np.fft.ifft(- self.k**2*u_hat).real
        else:
            lapl = np.dot(self.D2x, u.T).T
        return lapl

    def grad_lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft(u)
            d3 = np.fft.ifft(- 1j*self.k**3*u_hat).real
        else:
            d3 = np.dot(self.D3x, u.T).T
        return d3

    def grad_utility(self, phi, param):
        """Compute ∇ U = (∇πa + Γa ∇^3ρa) with πa = sum_b kappa_ab * rho_b for each species a."""
        pi = np.dot(param['kappa'],phi)
        dUdx  = self.grad(pi) + param['Gamma']*self.grad_lapl(phi)
        return dUdx
        
    def rhs_Vitelli(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""
        phi0 = 1- phi.sum(axis=0)
        dUdx = self.grad_utility(phi, param)
    
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        divJ = param['D']*phi0*self.lapl(phi) - param['D']*phi*self.lapl(phi0)-self.grad(phi*phi0*dUdx)
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            xi = np.random.normal(0, 1, size= phi.shape)
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_flux   = np.sqrt(2*param['D']*self.dx*rho_face/dt)*xi # Double check noise flux dx and dt dependence!!
            dnoise_dx    = self.grad(noise_flux)
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ

    def w0(self, phi, param):
        D = param["D"]
        beta = param["beta"]
        theta = param["theta"]
        Gamma = param["Gamma"]
        kappa = np.array([[theta-1,theta],[theta,theta-1]])
        pi = np.tensordot(kappa, phi + Gamma*self.lapl(phi), axes = (1,0))
        w0 = D/(1+np.exp(-beta*pi))
        gradw0 = D*beta/(2+2*np.cosh(beta*pi))*np.tensordot(kappa, self.grad(phi) + Gamma*self.grad_lapl(phi), axes = (1,0))
        laplw0 = self.grad(gradw0)
        return w0, gradw0, laplw0


    def rhs_Schelling(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""
        phi0   = 1- phi.sum(axis=0)
        w0, gradw0, laplw0 = self.w0(phi, param)
        
        """Compute grad J =   w0 (rho_0 ∂^2 rho -  rho ∂^2 rho_0)  + rho * rho_0 * ∂^2 w0 + 2 rho0 ∂ w0 . ∂ rho ) """
        divJ = w0*phi0*self.lapl(phi) - w0*phi*self.lapl(phi0)  + phi*phi0*laplw0 + 2*phi0*gradw0*self.grad(phi)
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            xi = np.random.normal(0, 1, size= phi.shape)
            rho_face     = np.maximum(w0*phi*phi0, 1e-14) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_flux   = np.sqrt(2*self.dx*rho_face/dt)*xi # Double check noise flux dx and dt dependence!!
            dnoise_dx    = self.grad(noise_flux)
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ

    def step(self, rhs, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            dphidt_pred = rhs(rho_pred, param,  dt, toggle_noise)
            rho_corr = phi + 0.5*dt*(dphidt + dphidt_pred)
            rho_next = rho_corr + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "RK4":
            k1 = dt * dphidt
            k2 = dt * rhs(phi + 1/2* k1, param, dt, toggle_noise)
            k3 = dt * rhs(phi + 1/2* k2, param, dt, toggle_noise)
            k4 = dt * rhs(phi + k3, param, dt, toggle_noise)
            rho_next = phi + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            
        
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
                        'D':     float: diffusion constant
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters
                        'Gamma': Coefficient of the lapl(phi) term in the utility
                        'b':     Reaction birth rate
                        'd':     Reaction death rate
                    "Schelling" with expected parameters in dictionary param:
                        'D':     float: diffusion constant
                        'theta': float: satisfaction threshold (between 0 and 1)
                        'Gamma': Coefficient of the lapl(phi) term in pi (sigma^2/2 of the Gaussian neighborhood kernel)
                        'beta':  Inverse temperature, large beta means scricter enforcement of threshold moves
                        'b':     Reaction birth rate
                        'd':     Reaction death rate
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
            else:
                raise ValueError("Unknown model, try 'Vitelli' or 'Schelling' ")
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {dissimilarity(phi_current):.6f}")
                phi_run[:,n//plot_every,:] = phi_current

        if verbatum:
            phi_diff = phi_run[0,:,:] - phi_run[1,:,:]
            im =plt.imshow(phi_diff, cmap = 'RdBu', aspect='auto', origin='lower', extent=[-self.L/2,self.L/2,0,dt*nsteps], vmin=-1, vmax=1)
            plt.title(fr"$D = {param['D']},\, \kappa aa = {param['kappa'][0,0]},\, \kappa ab = {param['kappa'][0,1]},\, \kappa ba = {param['kappa'][1,0]},\, \Gamma = { param['Gamma']}, \, b={param['b']}, \, d={param['d']} $")
            plt.xlabel("x")
            plt.ylabel("t")
            cbar = plt.colorbar(im, fraction=0.046)
            cbar.set_label(r"$\phi_a - \phi_b$",size=18)
            plt.show()
            
            plt.plot(self.x, phi_current[0], lw=2, color= 'C0', label = r'$\phi_a$')
            plt.plot(self.x, phi_current[1], lw=2, color= 'C1', label = r'$\phi_b$')
            plt.title("Final state")
            plt.ylim(0, 1.2)
            plt.xlabel('x')
            plt.ylabel('phi(x,t)')
            plt.legend()
            plt.show()
            
        return phi_run



class fhd_2d:
    '''Defines the 1-D fluctuating hydrodynamics class for simulating the sociohydrodynamic equations including noise and reactions:

    '''
    def __init__(self, L, N, bc = "periodic", fft = True):
        '''
        Initializes instance of the fhd class object

        Args:
            L:  tuple (Lx, Ly): spatial lengths of the domain, coordinates will be defined as running from -L/2 to L/2
            N:  tuple (Nx, Ny): number of discretization steps per coordinate
            bc: boundary conditions, choose among "periodic", "von Neumann" or "Dirichlet"
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
        elif bc == "von Neumann":
            raise ValueError("von Neumann boundary conditions not yet implemented")
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'von Neumann' or 'Dirichlet' ")
        
        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)*2*np.pi
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)*2*np.pi
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        self.phi_floor = 1e-14
        self.nspecies = 2

        # Matrixces Dx and Dy for 8-th order finite differences, needed for divergence function below
        self.Dx = makeD(self.Nx, self.dx)
        self.Dy = makeD(self.Ny, self.dy)
        if not fft:
            self.D2x = makeD2(self.Nx, self.dx)
            self.D3x = makeD3(self.Nx, self.dx)
            # self.D4x = makeD4(self.Nx, self.dx)
            self.D2y = makeD2(self.Ny, self.dy)
            self.D3y = makeD3(self.Ny, self.dy)
            # self.D4y = makeD4(self.Ny, self.dy)
        
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
            grad = np.zeros((2,)+u.shape)
            grad[0] = np.dot(self.Dx,u).transpose(1,0,2)
            grad[1] = np.dot(self.Dy,u.transpose(0,2,1)).transpose(1,2,0)
        return grad

        
    def lapl(self, u):
        if self.fft:
            u_hat = np.fft.fft2(u)
            
            lapl_hat = -(self.kx**2 + self.ky**2) * u_hat
            lapl = np.fft.ifft2(lapl_hat).real
        else:
            lapl = np.dot(self.D2x, u).transpose(1,0,2)
            lapl += np.dot(self.D2y,u.transpose(0,2,1)).transpose(1,2,0)             
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
            gradlap = np.zeros((2,)+u.shape)
            gradlap[0] = np.dot(self.D3x, u).transpose(1,0,2) + np.dot(self.Dx, np.dot(self.D2y, u.transpose(0,2,1)).transpose(1,2,0)).transpose(1,0,2)
            gradlap[1] = np.dot(self.D3y, u.transpose(0,2,1)).transpose(1,2,0) + np.dot(self.Dy, np.dot(self.D2x, u).transpose(1,2,0)).transpose(1,2,0)
        return gradlap

    def grad_utility(self, phi, param):
        """Compute ∇ U = (∇πa + Γa ∇^3ρa) with πa = sum_b kappa_ab * rho_b for each species a."""
        pi = np.tensordot(param['kappa'],phi, axes = (1,0))
        dUdx  = self.grad(pi) + param['Gamma']*self.grad_lapl(phi)
        return dUdx

    def div2d(self, vec):
        '''
        Divergence of a spatial vector: NOTE this uses finite differences instead of FFT, for some reason FFT gives bad results 
        when computing divergence of derivatives! To be understood properly...
        '''
        # vec_hat = np.fft.fft2(vec)
        
        # div = np.fft.ifft2(- 1j * self.kx * np.fft.fft2(vec[0])).real
        # div += np.fft.ifft2(- 1j * self.ky * np.fft.fft2(vec[1])).real
        div = np.dot(self.Dx, vec[0]).transpose(1,0,2)
        div += np.dot(self.Dy, vec[1].transpose(0,2,1)).transpose(1,2,0)
        return div
    
    def rhs_Vitelli(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""

        phi0   = 1- phi.sum(axis=0)
        phi0 = phi0.reshape((1,)+self.N)
        dUdx = self.grad_utility(phi, param)
        
    
        """Compute grad J =  D rho_0 ∂^2 rho - D rho ∂^2 rho_0  - ∂( rho*rho_0 ∂U_a) """
        rhorho0dUdx = phi*phi0*dUdx
        div_dUdx = self.div2d(rhorho0dUdx) 
        divJ = param['D']*phi0*self.lapl(phi) - param['D']*phi*self.lapl(phi0) - div_dUdx 
    
        """Generate stochastic flux term ∂x( rho ξ )"""
        if toggle_noise:
            xi = np.random.normal(0, 1, size= (2,)+phi.shape)
            rho_face     = np.maximum(phi*phi0, self.phi_floor**2) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_flux   = np.sqrt(2*param['D']*self.dx*self.dy*rho_face/dt)*xi # Double check noise flux dx and dt dependence!!
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ
        
    def w0(self, phi, param):
        D = param["D"]
        beta = param["beta"]
        theta = param["theta"]
        Gamma = param["Gamma"]
        kappa = np.array([[theta-1,theta],[theta,theta-1]])
        pi = np.tensordot(kappa, phi + Gamma*self.lapl(phi), axes = (1,0))
        w0 = D/(1+np.exp(-beta*pi))
        gradw0 = D*beta/(2+2*np.cosh(beta*pi))*np.tensordot(kappa, self.grad(phi) + Gamma*self.grad_lapl(phi), axes = (1,1)).transpose(1,0,2,3)
        laplw0 = self.div2d(gradw0)
        return w0, gradw0, laplw0

    def rhs_Schelling(self, phi, param, dt, toggle_noise = 1):
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
            rho_face     = np.maximum(w0*phi*phi0, 1e-14) # Changed to noise_floor^2 because phi*phi0 is also a square!
            noise_flux   = np.sqrt(2*self.dx*self.dy*rho_face/dt)*xi # Double check noise flux dx and dt dependence!!
            dnoise_dx    = self.div2d(noise_flux) 
        
            divJ += toggle_noise*dnoise_dx
        
        return divJ
        
    def step(self, rhs, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            rho_corr = phi + 0.5*dt*(dphidt + rhs(rho_pred, param,  dt, toggle_noise))
            rho_next = rho_corr + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        
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
                        'D':     float: diffusion constant
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters
                        'Gamma': Coefficient of the lapl(phi) term in the utility
                        'b':     Reaction birth rate
                        'd':     Reaction death rate
                    "Schelling" with expected parameters in dictionary param:
                        'D':     float: diffusion constant
                        'theta': float: satisfaction threshold (between 0 and 1)
                        'Gamma': Coefficient of the lapl(phi) term in pi (sigma^2/2 of the Gaussian neighborhood kernel)
                        'beta':  Inverse temperature, large beta means scricter enforcement of threshold moves
                        'b':     Reaction birth rate
                        'd':     Reaction death rate
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
            else:
                raise ValueError(f"Model {model} is unknown, please choose 'Vitelli' or 'Schelling'") 
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}, D_index = {dissimilarity(phi_current):.6f}")
                phi_run[:,n//plot_every,:,:] = phi_current

        if verbatum:
            phi_diff = phi_run[0,:,:,:] - phi_run[1,:,:,:]
            im =plt.imshow(phi_diff[-1], cmap = 'RdBu', aspect='auto', origin='lower', extent=[-self.Lx/2,self.Lx/2,-self.Ly/2,self.Ly/2], vmin=-1, vmax=1)
            plt.title(fr"$D = {param['D']},\, \kappa aa = {param['kappa'][0,0]},\, \kappa ab = {param['kappa'][0,1]},\, \kappa ba = {param['kappa'][1,0]},\, \Gamma = { param['Gamma']}, \, b={param['b']}, \, d={param['d']} $")
            plt.xlabel("x")
            plt.ylabel("t")
            cbar = plt.colorbar(im, fraction=0.046)
            cbar.set_label(r"$\phi_a - \phi_b$",size=14)
            plt.show()
            
        return phi_run

def dissimilarity(phi):
    phi0 = 1 - np.sum(phi,axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    D = np.mean(np.abs(phi[0] - global_dist[0]))/global_dist[0]
    D += np.mean(np.abs(phi[1] - global_dist[1]))/global_dist[1]
    D += np.mean(np.abs(phi0 - global_dist[2]))/global_dist[2]
    return D/2

def mean_relative_entropy(phi):
    phi0 = 1 - np.sum(phi, axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    
    
    # Ensure no division by zero or log of zero
    phi_combined = np.vstack([phi[0], phi[1], phi0]).reshape((3,)+ phi0.shape)
    global_dist = np.clip(global_dist, 1e-10, None)
    phi_combined = np.clip(phi_combined, 1e-10, None)
    Sglobal = - np.sum(global_dist * np.log(global_dist))
    
    kl_divergence = np.sum(phi_combined * np.log(phi_combined / global_dist.reshape((3,)+ len(phi0.shape)*(1,))), axis=0)
    mean_kl_divergence = np.mean(kl_divergence)/Sglobal
    
    
    return mean_kl_divergence

def makeD(Nx,dx):
    Dx = np.zeros((Nx, Nx))
    # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	
    for i in range(Nx):
        Dx[i, (i-4) % Nx] = 1/280
        Dx[i, (i-3) % Nx] = -4/105
        Dx[i, (i-2) % Nx] = 1/5
        Dx[i, (i-1) % Nx] = -4/5
        Dx[i, (i+1) % Nx] = 4/5
        Dx[i, (i+2) % Nx] = -1/5
        Dx[i, (i+3) % Nx] = 4/105
        Dx[i, (i+4) % Nx] = -1/280
    return Dx/dx

def makeD2(Nx, dx):
    D2x = np.zeros((Nx,Nx))
    # −1/560	8/315	−1/5	8/5	−205/72	8/5	−1/5	8/315	−1/560	
    for i in range(Nx):
        D2x[i, (i-4) % Nx] = -1/560
        D2x[i, (i-3) % Nx] = 8/315
        D2x[i, (i-2) % Nx] = -1/5
        D2x[i, (i-1) % Nx] = 8/5
        D2x[i,i] = -205/72
        D2x[i, (i+1) % Nx] = 8/5
        D2x[i, (i+2) % Nx] = -1/5
        D2x[i, (i+3) % Nx] = 8/315
        D2x[i, (i+4) % Nx] = -1/560
    return D2x/dx**2

def makeD3(Nx, dx):
    D3x = np.zeros((Nx,Nx))
    # −7/240	3/10	−169/120	61/30	0	−61/30	169/120	−3/10	7/240	
    for i in range(Nx):
        D3x[i, (i-4) % Nx] = -7/240
        D3x[i, (i-3) % Nx] = 3/10
        D3x[i, (i-2) % Nx] = -169/120
        D3x[i, (i-1) % Nx] = 61/30
        # D3[i,i] = 0
        D3x[i, (i+1) % Nx] = -61/30
        D3x[i, (i+2) % Nx] = 169/120
        D3x[i, (i+3) % Nx] = -3/10
        D3x[i, (i+4) % Nx] = 7/240
    return D3x/dx**3

def makeD4(Nx, dx):
    D4x = np.zeros((Nx,Nx))
    # 7/240	−2/5	169/60	−122/15	91/8	−122/15	169/60	−2/5	7/240	
    for i in range(Nx):
        D4x[i, (i-4) % Nx] = 7/240
        D4x[i, (i-3) % Nx] = -2/5
        D4x[i, (i-2) % Nx] = 169/60
        D4x[i, (i-1) % Nx] = -122/15
        D4x[i,i] = 91/8
        D4x[i, (i+1) % Nx] = -122/15
        D4x[i, (i+2) % Nx] = 169/60
        D4x[i, (i+3) % Nx] = -2/5
        D4x[i, (i+4) % Nx] = 7/240
    return D4x/dx**4
    