"""
1D and 2D implementation of 
∂t phi_a= ∂x [D phi_0 ∂x phi_a - D phi_a ∂x + phi_a phi_0 ∂x pi_a +  sqrt(2D dx^d phi_a phi_0) Z] + R(\phi)
with phi_0 = 1 - sum_a phi_a and Z white noise using a finite_differences discretization and 
a forward Euler time integrator. 

Features:
- Positivity floor for density
- multiplicative conservative noise sqrt(phi_a phi_0)
- R(\phi) implements a voter model with mean-field equation ∂t phi_a = phi*(b*pho_0 - d) with b and d birth and death rates respectively

Authors: Tuan Pham and Wout Merbis
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter


class fhd:
    '''Defines the 1-D fluctuating hydrodynamics class for simulating the sociohydrodynamic equations including noise and reactions:

    '''
    def __init__(self, L, N, bc = "periodic"):
        '''
        Initializes instance of the fhd class object

        Args:
            L:  length of the spatial domain, x coordinate will be defined as running from -L/2 to L/2
            N:  number of discretization steps
            bc: boundary conditions, choose among "periodic", "von Neumann" or "Dirichlet"
                (note: latter two are to be implemented
        '''
        
        self.L = L
        self.dx = L / N
        
        if bc == "periodic":
            self.N = N
            self.x = np.arange(-L/2, L/2, self.dx)
        elif bc == "von Neumann":
            raise ValueError("von Neumann boundary conditions not yet implemented")
        elif bc == "Dirichlet":
            raise ValueError("Dirichet boundary conditions not yet implemented")
        else:
            raise ValueError("Boundary conditions not properly specified, try: 'periodic', 'von Neumann' or 'Dirichlet' ")
        
        self.k = np.fft.fftfreq(N, d=self.dx)*2*np.pi
        self.phi_floor = 1e-10
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
        # if self.bc == "periodic":
        u_hat = np.fft.fft(u)
        grad = np.fft.ifft(1j*self.k*u_hat).real
        return grad
        
    def lapl(self, u):
        u_hat = np.fft.fft(u)
        lapl = np.fft.ifft(- self.k**2*u_hat).real
        return lapl

    def grad_lapl(self, u):
        u_hat = np.fft.fft(u)
        d3 = np.fft.ifft(- 1j*self.k**3*u_hat).real
        return d3

    def grad_utility(self, phi, param):
        """Compute ∇ U = (∇πa + Γa ∇^3ρa) with πa = sum_b kappa_ab * rho_b for each species a."""
        pi = np.dot(param['kappa'],phi)
        dUdx  = self.grad(pi) + param['Gamma']*self.grad_lapl(phi)
        return dUdx
        
    def rhs(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""
        phi0   = 1- phi.sum(axis=0)
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

    def step(self, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = self.rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            rho_corr = phi + 0.5*dt*(dphidt + self.rhs(rho_pred, param,  dt, toggle_noise))
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
            verbatum = True):
        ''' Runs the FHD simulation with specified parameters for nsteps, recording no_frames equally timed frames.

        Arg:
            phi:    Initial condition for phi, should have shape (nspecies, N)
            param:  Dictionary with parameter settings: Expected parameters:
                        'D':     float: diffusion constant
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters
                        'Gamma': Coefficient of the lapl(phi) term in the utility
                        'b':     Reaction birth rates
                        'd':     Reaction death rates
            nsteps: Number of simulation steps
            dt:     Size of the time step
            toggle_noise: strength of noise term, if zero, no noise is used
            no_frames: Number of frames saved in the final np. array
            scheme: Numerical integration scheme, choose between forward Euler 'FE' or predictor-corrector 'PC'
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
            phi_current = self.step(phi_current, param, dt, toggle_noise, scheme)
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}")
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
    def __init__(self, L, N, bc = "periodic"):
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
        
        self.phi_floor = 1e-10
        self.nspecies = 2

        # Matrixces Dx and Dy for 8-th order finite differences, needed for divergence function below
        self.Dx = np.zeros((self.Nx, self.Nx))
        # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	
        for i in range(self.Nx):
            self.Dx[i, (i-4) % self.Nx] = 1/280
            self.Dx[i, (i-3) % self.Nx] = -4/105
            self.Dx[i, (i-2) % self.Nx] = 1/5
            self.Dx[i, (i-1) % self.Nx] = -4/5
            self.Dx[i, (i+1) % self.Nx] = 4/5
            self.Dx[i, (i+2) % self.Nx] = -1/5
            self.Dx[i, (i+3) % self.Nx] = 4/105
            self.Dx[i, (i+4) % self.Nx] = -1/280
        self.Dx = self.Dx/self.dx
        
        self.Dy = np.zeros((self.Ny, self.Ny))
        # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	
        for i in range(self.Ny):
            self.Dy[i, (i-4) % self.Ny] = 1/280
            self.Dy[i, (i-3) % self.Ny] = -4/105
            self.Dy[i, (i-2) % self.Ny] = 1/5
            self.Dy[i, (i-1) % self.Ny] = -4/5
            self.Dy[i, (i+1) % self.Ny] = 4/5
            self.Dy[i, (i+2) % self.Ny] = -1/5
            self.Dy[i, (i+3) % self.Ny] = 4/105
            self.Dy[i, (i+4) % self.Ny] = -1/280
        self.Dy = self.Dy/self.dy
        
    def scale_down_pointwise(self, phi):
        ''' Scales down phi[a] on sites where sum_a phi[a] < 1
        '''
        sumphi = np.sum(phi, axis=0)
        divisor = np.maximum(sumphi+self.phi_floor,1)
        # scale down phi only at point where sum exceeds one
        phi = phi/divisor
        return phi

    def grad(self, u):
        u_hat = np.fft.fft2(u)
    
        grad_x_hat = 1j * self.kx * u_hat
        grad_y_hat = 1j * self.ky * u_hat
    
        grad = np.zeros((2,)+u.shape)
        grad[0] = np.fft.ifft2(grad_x_hat).real
        grad[1] = np.fft.ifft2(grad_y_hat).real
        return grad

        
    def lapl(self, u):
        u_hat = np.fft.fft2(u)
        
        lapl_hat = -(self.kx**2 + self.ky**2) * u_hat
        lapl = np.fft.ifft2(lapl_hat).real
        return lapl

    def grad_lapl(self, u):
        u_hat = np.fft.fft2(u)
        
        gradlap_x_hat = - 1j * self.kx * (self.kx**2 + self.ky**2) * u_hat
        gradlap_y_hat = - 1j * self.ky * (self.kx**2 + self.ky**2) * u_hat
    
        gradlap = np.zeros((2,)+u.shape)
        gradlap[0] = np.fft.ifft2(gradlap_x_hat).real
        gradlap[1] = np.fft.ifft2(gradlap_y_hat).real
        return gradlap

    # Do not use!
    # def lapl_squared(self, u):
    #     u_hat = np.fft.fft2(u)
        
    #     lap_sq_hat = ((self.kx**2 + self.ky**2)**2) * u_hat
    #     lap_sq = np.fft.ifft2(lap_sq_hat).real
    #     return lap_sq

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
    
    def rhs(self, phi, param, dt, toggle_noise = 1):
        """Compute RHS of the equation"""

        phi0   = 1- phi.sum(axis=0)
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

    def step(self, phi, param, dt, toggle_noise, scheme):
        phi_tot = np.sum(phi, axis=0)
        dphidt = self.rhs(phi, param, dt, toggle_noise)
        rho_pred = phi + dt * dphidt
    
        if scheme == "FE":
            rho_next = rho_pred + dt * phi*(param['b']*(1-phi_tot) - param['d'])
        elif scheme == "PC":
            rho_corr = phi + 0.5*dt*(dphidt + self.rhs(rho_pred, param,  dt, toggle_noise))
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
            verbatum = True):
        ''' Runs the FHD simulation with specified parameters for nsteps, recording no_frames equally timed frames.

        Arg:
            phi:    Initial condition for phi, should have shape (nspecies, N)
            param:  Dictionary with parameter settings: Expected parameters:
                        'D':     float: diffusion constant
                        'kappa': numpy array of shape (nspecies, nspecies) with linear utility parameters
                        'Gamma': Coefficient of the lapl(phi) term in the utility
                        'b':     Reaction birth rates
                        'd':     Reaction death rates
            nsteps: Number of simulation steps
            dt:     Size of the time step
            toggle_noise: strength of noise term, if zero, no noise is used
            no_frames: Number of frames saved in the final np. array
            scheme: Numerical integration scheme, choose between forward Euler 'FE' or predictor-corrector 'PC'
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
            phi_current = self.step(phi_current, param, dt, toggle_noise, scheme)
        
            if n % plot_every == 0:
                if verbatum:
                    print(f"Step {n}/{nsteps}: mean rho = {phi_current.mean():.6f}, min = {phi_current.min():.6e}")
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



# Dx = np.zeros((Nx, Nx))
# # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	
# for i in range(Nx):
#     Dx[i, (i-4) % Nx] = 1/280
#     Dx[i, (i-3) % Nx] = -4/105
#     Dx[i, (i-2) % Nx] = 1/5
#     Dx[i, (i-1) % Nx] = -4/5
#     Dx[i, (i+1) % Nx] = 4/5
#     Dx[i, (i+2) % Nx] = -1/5
#     Dx[i, (i+3) % Nx] = 4/105
#     Dx[i, (i+4) % Nx] = -1/280
# Dx = Dx/dx

# Dy = np.zeros((Ny, Ny))
# # 1/280	−4/105	1/5	−4/5	0	4/5	−1/5	4/105	−1/280	
# for i in range(Ny):
#     Dy[i, (i-4) % Ny] = 1/280
#     Dy[i, (i-3) % Ny] = -4/105
#     Dy[i, (i-2) % Ny] = 1/5
#     Dy[i, (i-1) % Ny] = -4/5
#     Dy[i, (i+1) % Ny] = 4/5
#     Dy[i, (i+2) % Ny] = -1/5
#     Dy[i, (i+3) % Ny] = 4/105
#     Dy[i, (i+4) % Ny] = -1/280
# Dy = Dy/dy

# D2x = np.zeros((Nx,Nx))
# # −1/560	8/315	−1/5	8/5	−205/72	8/5	−1/5	8/315	−1/560	
# for i in range(Nx):
#     D2x[i, (i-4) % Nx] = -1/560
#     D2x[i, (i-3) % Nx] = 8/315
#     D2x[i, (i-2) % Nx] = -1/5
#     D2x[i, (i-1) % Nx] = 8/5
#     D2x[i,i] = -205/72
#     D2x[i, (i+1) % Nx] = 8/5
#     D2x[i, (i+2) % Nx] = -1/5
#     D2x[i, (i+3) % Nx] = 8/315
#     D2x[i, (i+4) % Nx] = -1/560
# D2x = D2x/dx**2

# D2y = np.zeros((Ny,Ny))
# # −1/560	8/315	−1/5	8/5	−205/72	8/5	−1/5	8/315	−1/560	
# for i in range(Ny):
#     D2y[i, (i-4) % Ny] = -1/560
#     D2y[i, (i-3) % Ny] = 8/315
#     D2y[i, (i-2) % Ny] = -1/5
#     D2y[i, (i-1) % Ny] = 8/5
#     D2y[i,i] = -205/72
#     D2y[i, (i+1) % Ny] = 8/5
#     D2y[i, (i+2) % Ny] = -1/5
#     D2y[i, (i+3) % Ny] = 8/315
#     D2y[i, (i+4) % Ny] = -1/560
# D2y = D2y/dy**2

# D3x = np.zeros((Nx,Nx))
# # −7/240	3/10	−169/120	61/30	0	−61/30	169/120	−3/10	7/240	
# for i in range(Nx):
#     D3x[i, (i-4) % Nx] = -7/240
#     D3x[i, (i-3) % Nx] = 3/10
#     D3x[i, (i-2) % Nx] = -169/120
#     D3x[i, (i-1) % Nx] = 61/30
#     # D3[i,i] = 0
#     D3x[i, (i+1) % Nx] = -61/30
#     D3x[i, (i+2) % Nx] = 169/120
#     D3x[i, (i+3) % Nx] = -3/10
#     D3x[i, (i+4) % Nx] = 7/240
# D3x = D3x/dx**3

# D3y = np.zeros((Ny,Ny))
# # −7/240	3/10	−169/120	61/30	0	−61/30	169/120	−3/10	7/240	
# for i in range(Ny):
#     D3y[i, (i-4) % Ny] = -7/240
#     D3y[i, (i-3) % Ny] = 3/10
#     D3y[i, (i-2) % Ny] = -169/120
#     D3y[i, (i-1) % Ny] = 61/30
#     # D3[i,i] = 0
#     D3y[i, (i+1) % Ny] = -61/30
#     D3y[i, (i+2) % Ny] = 169/120
#     D3y[i, (i+3) % Ny] = -3/10
#     D3y[i, (i+4) % Ny] = 7/240
# D3y = D3y/dy**3


# def grad(u, dx):
#     global bc, k
#     # Slower:
#     # grad = (-np.roll(u,-2,axis=-1)/12+2*np.roll(u,-1,axis=-1)/3-2*np.roll(u,1,axis=-1)/3+np.roll(u,2,axis=-1)/12)
#     # grad = np.dot(Dx, u.T).T
#     u_hat = np.fft.fft(u)
#     grad = np.fft.ifft(1j*k*u_hat).real
#     if bc == "vN":
#         if len(grad.shape) == 1:
#             grad = grad.reshape(1, len(grad))
#             u = u.reshape(1,len(u))
#         # grad[i] = (u[i-2] - 8*u[i-1] + 8*u[i+1] - u[i+2])/12 and at the boundaries reflect u[-dx] = u[dx]!
#         grad[:,0] = 0
#         grad[:,1] = (u[:,1] - 8*u[:,0] + 8*u[:,2] - u[:,3])/(12*dx)
#         grad[:,-1] = 0
#         grad[:,-2] = (u[:,-4] - 8*u[:,-3] - u[:,-2] + 8*u[:,-1]) / (12*dx)
#         if grad.shape[0] == 1:
#             grad = grad.reshape(grad.shape[1])
#             u = u.reshape(u.shape[1])
#     return grad

# # def grad2d(u):
# #     global bc, Dx, Dy
# #     grad = np.zeros((2,)+u.shape)
# #     grad[0] = np.dot(Dx,u).transpose(1,0,2)
# #     grad[1] = np.dot(Dy,u.transpose(0,2,1)).transpose(1,2,0)
# #     return grad

# def lapl(u, dx):
#     global bc, k
#     # lapl = -np.roll(u,-2,axis=-1)/12 + (4/3)*np.roll(u,-1,axis=-1) - (5/2)*u + (4/3)*np.roll(u,1,axis=-1) - (1/12)*np.roll(u,2,axis=-1)
#     # lapl = np.dot(D2x,u.T).T
    
#     u_hat = np.fft.fft(u)
#     lapl = np.fft.ifft(- k**2*u_hat).real
    
#     if bc == "vN":
#         if len(lapl.shape) == 1:
#             lapl = lapl.reshape(1, len(lapl))
#             u = u.reshape(1,len(u))
#         # lapl[i] = (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2] )/12 then reflect u[-dx] = u[dx] and u[-2dx] = u[2dx]     
#         lapl[:,0] = (- 30*u[:,0] + 2*16*u[:,1] - 2*u[:,2]) / (12*dx**2)
#         lapl[:,1] = (16*u[:,0] - (30+1)*u[:,1] + 16*u[:,2] - u[:,3]) / (12*dx**2)
#         lapl[:,-1] = ( -30*u[:,-1] + 2*16*u[:,-2] - 2*u[:,-3]) / (12*dx**2)
#         lapl[:,-2] = (16*u[:,-1] - (30+1)*u[:,-2] + 16*u[:,-3] - u[:,-4]) / (12*dx**2)
#         if lapl.shape[0] == 1:
#             lapl = lapl.reshape(lapl.shape[1])
#             u = u.reshape(u.shape[1])          
#     return lapl

# # def lapl2d(u):
# #     global bc, D2x, D2y
# #     lapl = np.dot(D2x, u).transpose(1,0,2)
# #     lapl += np.dot(D2y,u.transpose(0,2,1)).transpose(1,2,0) 
# #     return lapl

# def grad_lapl(u, dx):
#     global bc, D3x
#     # d3 = -(1/8)*np.roll(u,-3,axis=-1) + np.roll(u,-2,axis=-1) - (13/8)*np.roll(u,-1,axis=-1) + (13/8)*np.roll(u,1,axis=-1) - np.roll(u,2,axis=-1) + (1/8)*np.roll(u,3,axis=-1)
#     # d3 = np.dot(D3x, u.T).T
#     u_hat = np.fft.fft(u)
#     d3 = np.fft.ifft(- 1j*k**3*u_hat).real
#     if bc == "vN":
#         if len(d3.shape) == 1:
#             d3 = d3.reshape(1, len(d3))
#             u = u.reshape(1,len(u))
#         d3[:,0] = 0
#         d3[:,1] = (-u[:,4] + 8*u[:,3] - (13-1)*u[:,2] + (0-8)*u[:,1] + 13*u[:,0]) / (8) /dx**3
#         d3[:,2] = (-u[:,5] + 8*u[:,4] - 13*u[:,3] + 0*u[:,2] + (13+1)*u[:,1]-8*u[:,0]) / (8)/dx**3
#         d3[:,-1] = 0
#         d3[:,-2] = (u[:,-5] - 8*u[:,-4] + (13-1)*u[:,-3] +(0+8)*u[:,-2]- 13*u[:,-1] ) / (8)/dx**3
#         d3[:,-3] = (u[:,-6] - 8*u[:,-5] + (13)*u[:,-4] +(0)*u[:,-3]- (13+1)*u[:,-2]+8*u[:,-1] ) / (8)/dx**3
#         if d3.shape[0] == 1:
#             d3 = d3.reshape(d3.shape[1])    
#             u = u.reshape(u.shape[1])
#     return d3

# # def grad_lapl2d(u):
# #     global bc, D3x, D3y, D2x, D2y, Dx, Dy
# #     gradlap = np.zeros((2,)+u.shape)
# #     gradlap[0] = np.dot(D3x, u).transpose(1,0,2) + np.dot(Dx, np.dot(D2y, u.transpose(0,2,1)).transpose(1,2,0)).transpose(1,0,2)
# #     gradlap[1] = np.dot(D3y, u.transpose(0,2,1)).transpose(1,2,0) + np.dot(Dy, np.dot(D2x, u).transpose(1,2,0)).transpose(1,2,0)
# #     return gradlap