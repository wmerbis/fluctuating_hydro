# Fluctuating Hydrodynamics

Code base for simulating fluctuating stochastic reaction diffusion systems in one and two dimensions with applications to socio-economic modeling of segregation and inequality

1D and 2D implementation of the equation:\
$$\partial_t \phi_a= \nabla (D \phi_0 \nabla \phi_a - D \phi_a \nabla \phi_0 + \phi_a \phi_0 \nabla \pi_a - J_v +  \sqrt{2D h^2 \phi_a \phi_0} Z) + \sqrt{4D_v \phi_A \phi_B} W^a  $$ \
with $\phi_0 = 1 - \sum_a \phi_a$ and $Z$ white noise using a finite_differences discretization and a forward Euler time integrator. 

Features:
- Positivity floor for density
- multiplicative conservative noise $\sim \sqrt{\phi_a \phi_0}$
- $\pi_a$ is a local utility function $\pi_a = \sum_b \kappa^{ab} \phi_b + \Gamma^{ab} \nabla^2 \phi_b $
- $J_v$ implements the voter model current
- non-conservative demographic noise $\sim \sqrt{\phi_a \phi_b}$ with $W^A = - W^B =$ Gaussian white noise.

This code base is work in progress. Do not desimate without explicit permission of the authors.

Authors: Tuan Pham and Wout Merbis
