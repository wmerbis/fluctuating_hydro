# Conservative bound-preserving limiter

`SchellingVoterFVSolver` supports three limiter modes:

- `limiter_mode="none"`: do not enforce the cellwise simplex during time stepping.
- `limiter_mode="clip"`: use the historical local positivity/simplex projection.
- `limiter_mode="conservative"`: use the conservative flux-based bound-preserving limiter for the fully active periodic Cartesian explicit/stochastic update.

The conservative limiter is intended as the long-term fix for projection-induced occupied-mass drift.  Instead of clipping updated cells independently, it assembles the deterministic finite-volume and conservative Schelling-noise increments as shared face increments.  Each cell computes admissible budgets for species outflow and occupied-density inflow, and every face is rescaled by one coefficient shared by its two adjacent cells.  Because the same limited face increment is added to one cell and subtracted from the other, `M_A`, `M_B`, and `M_occ` retain the finite-volume conservation structure up to roundoff.  Voter noise is treated as a local `A <-> B` exchange and is clipped only to the interval that keeps both species nonnegative, preserving occupied mass.

This mode is slower than `clip` because it needs extra face arrays, per-cell flux budgets, and a second pass over all faces before committing the update.  The cost is most worthwhile for long stochastic runs, sharp interfaces, large time steps, or high-occupancy regimes where the old clip projection is activated often.  In those regimes independent local clipping can remove negative species density or rescale overfull cells without compensating neighboring cells, which creates the occupied-mass drift diagnosed in long-run tests.

Current scope: the conservative flux limiter is implemented for the regular fully active periodic Cartesian explicit/stochastic path.  If the optional Fourier Gamma step is enabled, any post-Gamma bound violations are handled by a slower global conservative redistribution that preserves species masses; this residual path is reported separately in limiter diagnostics.  Masked/no-flux grids should use `clip` for comparison or `none` for diagnostics until a face-based limiter tailored to those operators is added.
