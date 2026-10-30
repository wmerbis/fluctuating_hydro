# Schelling reciprocal/non-reciprocal hydrodynamic sweep

Place `schelling_phase_sweep.py` in the repository root, alongside `fhd/`.
It expects your **sigma-patched** `fhd/FHD_2D.py` and the previously installed
`fhd/observables.py`. No modification to the `fhd` package is needed.

## Quick checks

```bash
python schelling_phase_sweep.py --dry-run
python schelling_phase_sweep.py --dt-test --dt-test-beta 3 --dt-test-x -1 --dt-test-time 0.5
python schelling_phase_sweep.py --out sweep_smoke --n-grid 3 --nsteps 100 --frames 10 --max-jobs 3 --workers 2 --no-dynamics
```

The last command runs 3 points and is deliberately shorter than the full sweep.

## Full 21×21 scans (882 runs total)

```bash
python schelling_phase_sweep.py --workers 4 --out phase_sweep
```

`--workers` / `--n-cores` controls multiprocessing. Default physical parameters:

- N=(128,128), L=(50,50), Neumann; **21 beta points 0..3**.
- Reciprocal: kappa=[[1,x],[x,1]], x in [-1,1] (21 points).
- Nonreciprocal: kappa=[[1,x],[-x,1]], x in [0,1] (21 points).
- D=(1,1), sigma=.5, cell capacity=400, h=sqrt(dx*dy/400)=.01953125.
- Gaussian finite-volume noise enabled, midpoint Theta, symmetric face mobility.
- FE, dt=.001, 10,000 steps, 500 **intervals**, returning 501 frames.
- Homogeneous rho_A=rho_B=.35 plus independent Gaussian perturbations std=.01.
- Starting seeds vary deterministically across parameter points.

**This is a substantial sweep**. Use the short smoke run and the deterministic
`--dt-test` before the full sweep. The conservative bounded-hopping sufficient
positivity condition is about dt <= .019 for this grid, so .001 passes that
necessary preliminary check; it **does not** prove time-step convergence of
nonlinear fluctuations or pattern formation. Compare dt and dt/2 for a few
points near transitions and at the strongest segregation/migration.

### Disk and memory

Each run uses roughly 131 MB in memory for `phi_run` alone (float64) and
additional arrays for Fourier analysis. Four workers can consume several GB.
The default output saves only a few density snapshots, time series of order
parameters, DCT radial spectra, and (downsampled/low-k) temporal spectra.
It does **not** save the full 501-frame density trajectory. To save full
trajectories, add `--save-trajectory` (large disk cost).

Temporal diagnostics default to k<=3 and every fourth saved frame, after the
50% burn-in. These choices can affect frequency resolution and aliasing:
`--dynamic-kmax`, `--dynamic-stride`, `--max-lag`, `--burn-fraction` are
configurable. For a richer migration analysis, use a smaller temporal stride.

You can restrict excess order parameters to a physically meaningful k band:
`--k-band 0 3`. For phase classification this is often better than including
all grid-scale fluctuation power. The *full* radial DCT spectrum is saved in
all cases. `C` is NaN when the excess power is too weak to identify a channel.

### Results and resuming

- `phase_sweep/reciprocal/bXX_xXX_r00.npz` and
  `phase_sweep/nonreciprocal/bXX_xXX_r00.npz` store per-run results.
- `phase_sweep/summary.csv` stores scalar late-time averages and variances.
- `phase_sweep/reciprocal_grids.npz` and `nonreciprocal_grids.npz` store
  rectangular parameter heatmaps, indexed `[beta_index, x_index]`.
- `phase_sweep/failures/*.txt` stores individual failed run traces.
- Re-running the same command skips completed points. Changed physical
  parameters require a different `--out` or explicit `--overwrite`, which
  invalidates incompatible old outputs. `--aggregate-only` rebuilds summary.

Plot the results with:

```bash
python plot_schelling_phase_sweep.py --out phase_sweep --save phase_sweep.png
```

### Selected NPZ keys

`times`, `snapshots`, `snapshot_indices`, `k`, `S_dct` (T, kbin, 2, 2),
`S_plus`, `S_minus`, `S_reference`, `S_excess`, `P_plus`, `P_minus`, `C`,
`I_seg`, `I_seg_excess`, `I_occ`, `I_occ_excess`, `dissimilarity`, `entropy_kl`,
`normalized_entropy_segregation`, `local_shannon_entropy`,
`legacy_dissimilarity`, `legacy_mean_relative_entropy`, `mean_m`, `variance_m`,
`mean_n`, `variance_n`, `mean_vacancy`, `variance_vacancy`, `total_vacancy`,
`C_k_tau`, `tau`, `S_k_omega`, `omega`, `diag_*`.

`S_dct` is an orthonormal DCT-II **radial covariance matrix** for Neumann
boundaries. It has the spatial normalization of `fhd.observables.py`.
`dissimilarity` uses 0.5 sum_i |rhoA_i/N_A - rhoB_i/N_B|.
`entropy_kl` is occupancy-weighted KL(local species mix || global mix),
`normalized_entropy_segregation` divides by global composition entropy,
`local_shannon_entropy` measures the local binary composition entropy.
The `legacy_*` metrics use exactly the installed `fhd.operations` functions
when available; otherwise they remain NaN, while independent metrics are
still computed.

**Important:** periodic Fourier modes and Neumann DCT modes have different
symmetries. A ring-averaged DCT spatiotemporal spectrum can detect temporal
oscillations but is not by itself a directional traveling-wave measurement.
For migration use time-lagged spatial cross-correlations or directed modes
from a suitable analysis of the full saved trajectory.
