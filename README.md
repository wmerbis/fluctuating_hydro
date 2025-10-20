# Fluctuating Hydrodynamics

Code base for simulating fluctuating stochastic reaction diffusion systems in one and two dimensions with applications to socio-economic modeling of segregation and inequality

1D and 2D implementation of the equation:\
$$\partial_t \phi_a= \nabla (D \phi_0 \nabla \phi_a - D \phi_a \nabla \phi_0 + \phi_a \phi_0 \nabla \pi_a +  \sqrt{2D dx^d \phi_a \phi_0} Z) + R(\phi)$$ \
with $\phi_0 = 1 - \sum_a \phi_a$ and $Z$ white noise using a finite_differences discretization and a forward Euler time integrator. 

Features:
- Positivity floor for density
- multiplicative conservative noise $\sim \sqrt{\phi_a \phi_0}$
- $R(\phi)$ implements a voter model with mean-field equation $\partial_t \phi_a = \phi*(b*\phi_0 - d)$ with $b$ and $d$ birth and death rates respectively

This code base is work in progress. Do not desimate without explicit permission of the authors.

Authors: Tuan Pham and Wout Merbis
