"""One-step conditional-moment benchmark for the *new* Schelling face reactions.

Place this at your repository root beside package fhd/, and run:
    python schelling_one_step_benchmark.py --help

This intentionally tests YOUR INSTALLED solver, not a reimplementation of its
random-number algorithm. Each initial field is held fixed across independent
noise realizations, then repeated for several independently generated fields.

Outputs conditional mean increment E[delta rho | phi(0)] and conditional
Var(delta rho | phi(0)), compared against two independent-generator targets:
  ideal : all the matched directed hopping propensities (no threshold),
  gated : propensities after current face sampler's epsilon availability test.
Both targets are infinitesimal-time / first-order-in-dt predictions, so finite
dt color splitting can produce deviations even when the sampler is correct.

Default N=(128,128), L=(1,1), dt=1e-5. Large h may imply epsilon>rho0
and shut off all reaction events; this is explicitly reported.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np

try:
    from fhd.FHD_2D import fhd_2d, schelling_theta
except ImportError as exc:
    raise ImportError('Run from repository root, with fhd/FHD_2D.py and fhd/operations.py importable.') from exc


def initial_field(N, rho, perturb, rng):
    phi = np.broadcast_to(np.asarray(rho, dtype=float)[:, None, None], (2, *N)).copy()
    if perturb:
        jitter = rng.uniform(-perturb, perturb, size=phi.shape)
        jitter -= jitter.mean(axis=(1, 2), keepdims=True)
        phi += jitter
    if np.min(phi) < 0 or np.max(phi.sum(axis=0)) >= 1:
        raise ValueError('initial field outside simplex; reduce --perturb')
    return phi


def instantaneous_moments(solver, phi, param, dt):
    """Predict moment maps of a 1-step reaction from its initial generators.

    Returns ideal and availability-gated drift/diagonal variance for BOTH
    species, plus fraction of blocked initial propensity. The spatial
    divergences and variance sums use actual periodic/Neumann bond geometry.
    The propensity is D*Omega/delta² * rho_source * vac_target * (1±s).
    """
    cache = solver.prepare_schelling(phi, param)
    omega = solver.cell_area / cache['h']**2
    eps = 1.0 / omega
    coeff, gamma = cache['coeff'], cache['Gamma']
    D = np.asarray(param['D'], dtype=float)
    vac = 1.0 - phi.sum(axis=0)
    lap = solver.lapl(phi)
    mode = param.get('theta_face', 'midpoint')
    mean_ideal = np.zeros_like(phi); var_ideal = np.zeros_like(phi)
    mean_gated = np.zeros_like(phi); var_gated = np.zeros_like(phi)
    attempted, rejected = 0.0, 0.0

    for axis, dx in ((1, solver.dx), (2, solver.dy)):
        if solver.bc == 'periodic':
            left = [slice(None)]*3; right = [slice(None)]*3
            rL, rR = phi, np.roll(phi, -1, axis=axis)
            vL, vR = vac, np.roll(vac, -1, axis=axis-1)
            lapL, lapR = lap, np.roll(lap, -1, axis=axis)
            # For a periodic bond, the left endpoint is each cell, the right
            # endpoint is the neighbor at +axis. Scatter back via np.roll.
            def add(field, at_left, at_right):
                field += at_left + np.roll(at_right, 1, axis=axis)
        elif solver.bc == 'Neumann':
            ls = [slice(None)]*3; rs = [slice(None)]*3
            ls[axis], rs[axis] = slice(None, -1), slice(1, None)
            ls, rs = tuple(ls), tuple(rs)
            rL, rR = phi[ls], phi[rs]
            vL, vR = vac[ls[1:]], vac[rs[1:]]
            lapL, lapR = lap[ls], lap[rs]
            def add(field, at_left, at_right):
                field[ls] += at_left; field[rs] += at_right
        else:
            raise ValueError('Supported BCs: periodic, Neumann')

        drho = rR - rL
        if mode == 'midpoint':
            theta = schelling_theta(0.5*(rL+rR), coeff)
        elif mode == 'endpoints':
            theta = 0.5*(schelling_theta(rL, coeff) + schelling_theta(rR, coeff))
        elif mode == 'gauss3':
            z = np.sqrt(15.0)/10.0
            theta = (5*schelling_theta(rL+(0.5-z)*drho, coeff) + 8*schelling_theta(0.5*(rL+rR), coeff)
                    + 5*schelling_theta(rL+(0.5+z)*drho, coeff))/18.0
        else:
            raise ValueError('Unknown theta_face')
        dlap = lapR - lapL

        for a in range(2):
            u = 0.5*(10.0*(theta[a,0]*drho[0] + theta[a,1]*drho[1])
                     + gamma[a,0]*dlap[0] + gamma[a,1]*dlap[1])
            s = np.tanh(u)
            pref = D[a]*omega/dx**2
            fwd = pref*rL[a]*vR*(1.0+s)
            bwd = pref*rR[a]*vL*(1.0-s)
            fwd_g = fwd*((rL[a] >= eps) & (vR >= eps))
            bwd_g = bwd*((rR[a] >= eps) & (vL >= eps))
            attempted += (fwd+bwd).sum()
            rejected += (fwd+bwd-fwd_g-bwd_g).sum()

            # One event transfers epsilon density: L loses for fwd, gains for bwd.
            def contribute(mean, var, plus, minus):
                mu = eps*dt*(minus-plus)
                vv = eps**2*dt*(plus+minus)
                # Shape (2,Nx,Ny), assign to correct species only.
                if solver.bc == 'periodic':
                    mean[a] += mu + np.roll(-mu, 1, axis=axis-1)
                    var[a] += vv + np.roll(vv, 1, axis=axis-1)
                else:
                    mean[a][ls[1:]] += mu
                    mean[a][rs[1:]] -= mu
                    var[a][ls[1:]] += vv
                    var[a][rs[1:]] += vv
            contribute(mean_ideal, var_ideal, fwd, bwd)
            contribute(mean_gated, var_gated, fwd_g, bwd_g)

    return dict(mean_ideal=mean_ideal, var_ideal=var_ideal, mean_gated=mean_gated, var_gated=var_gated,
                blocked_rate_fraction=(rejected/attempted if attempted else np.nan), eps=eps, omega=omega)


def benchmark_case(solver, phi0, param, dt, nrep, random_seed):
    theory = instantaneous_moments(solver, phi0, param, dt)
    work = solver._ensure_work(phi0.dtype)
    # Important: cache preparation is done once for fixed initial fields.
    # Store sufficient statistics rather than recording all trajectories.
    sum_inc = np.zeros_like(phi0); sum_sq = np.zeros_like(phi0)
    masses = np.sum(phi0, axis=(1, 2))
    min_density, min_vacancy, max_mass_error = 1., 1., 0.
    for rep in range(nrep):
        solver.set_seed(random_seed+rep)
        phi1 = solver.step(solver.rhs_Schelling, phi0, param, dt, True, 'FE', work=work, step=0)
        incr = phi1 - phi0
        sum_inc += incr; sum_sq += incr*incr
        min_density = min(min_density, phi1.min())
        min_vacancy = min(min_vacancy, (1.0 - phi1.sum(axis=0)).min())
        max_mass_error = max(max_mass_error, np.max(np.abs(phi1.sum(axis=(1,2))-masses)))
    emp_mean = sum_inc / nrep
    emp_var = (sum_sq - sum_inc*sum_inc/nrep)/(nrep-1)
    out = dict(min_density=min_density, min_vacancy=min_vacancy, max_mass_error=max_mass_error)
    for a, label in enumerate(('A', 'B')):
        mv_i = theory['mean_ideal'][a]; mv_g = theory['mean_gated'][a]
        vi = theory['var_ideal'][a]; vg = theory['var_gated'][a]
        ev = emp_var[a]; em = emp_mean[a]
        out.update({
            f'{label}_mean_rms_theory_ideal': float(np.sqrt(np.mean(mv_i**2))),
            f'{label}_mean_rms_residual_ideal': float(np.sqrt(np.mean((em-mv_i)**2))),
            f'{label}_mean_rms_residual_gated': float(np.sqrt(np.mean((em-mv_g)**2))),
            f'{label}_mean_rms_sampling_SE_gated': float(np.sqrt(np.mean(vg)/nrep)),
            f'{label}_variance_measured': float(np.mean(ev)),
            f'{label}_variance_ideal': float(np.mean(vi)),
            f'{label}_variance_gated': float(np.mean(vg)),
            f'{label}_variance_ratio_ideal': float(np.mean(ev)/np.mean(vi)) if np.mean(vi)>0 else np.nan,
            f'{label}_variance_ratio_gated': float(np.mean(ev)/np.mean(vg)) if np.mean(vg)>0 else np.nan,
        })
    out.update(omega=theory['omega'], epsilon=theory['eps'], blocked_rate_fraction=theory['blocked_rate_fraction'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', nargs=2, type=int, default=(128,128)); p.add_argument('--L', nargs=2, type=float, default=(1.,1.))
    p.add_argument('--dt', type=float, default=1e-5)
    p.add_argument('--h', nargs='+', type=float, default=[.0005,.001,.002,.004,.005])
    p.add_argument('--beta', nargs='+', type=float, default=[0.0, .63, 5.0])
    p.add_argument('--rho', nargs=2, type=float, default=(.35,.35))
    p.add_argument('--D', nargs=2, type=float, default=(1.,1.))
    p.add_argument('--kappa', nargs=4, type=float, default=(.6,-.4,-.4,.6))
    p.add_argument('--initial-conditions', type=int, default=2)
    p.add_argument('--replicates', type=int, default=32)
    p.add_argument('--perturb', type=float, default=.01)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--theta-face', choices=('midpoint','endpoints','gauss3'), default='midpoint')
    p.add_argument('--bc', choices=('periodic','Neumann'), default='periodic')
    p.add_argument('--fft', action='store_true', help='Use FFT for stiffness Laplacian instead of 8th-order finite differences')
    p.add_argument('--csv', default='schelling_one_step_results.csv')
    args = p.parse_args()
    if args.replicates < 2 or args.initial_conditions < 1:
        p.error('replicates >= 2 and initial-conditions >= 1 required')
    kapp = np.asarray(args.kappa).reshape(2,2)
    solver = fhd_2d(L=tuple(args.L), N=tuple(args.N), bc=args.bc, fft=args.fft,
                    schelling_flux='face_reaction', projection_mode='redistribute', reaction_diagnostic=False)
    rng = np.random.default_rng(args.seed)
    rows = []; tic = time.perf_counter()
    print('One-step face-reaction benchmark: N={}, L={}, dt={}, nIC={}, replicas={}'.format(
          args.N,args.L,args.dt,args.initial_conditions,args.replicates))
    print('D*dt/dx^2 =', max(args.D)*args.dt/(solver.dx**2), '; compare against 1 for splitting refinement')
    print('Ideal = all hopping rates; gated = rates remaining after epsilon availability test.')
    for beta in args.beta:
        for h in args.h:
            pset = dict(D=np.asarray(args.D), beta=beta, kappa=kapp, rho_bar=np.asarray(args.rho),
                        h=h, theta_face=args.theta_face)
            for ic in range(args.initial_conditions):
                phi0 = initial_field(tuple(args.N), args.rho, args.perturb, rng)
                row = benchmark_case(solver, phi0, pset, args.dt, args.replicates,
                                     args.seed + 100000*len(rows))
                row.update(beta=beta,h=h,ic=ic,dt=args.dt,Nx=args.N[0],Ny=args.N[1])
                rows.append(row)
                print('beta={:5g} h={:.5g} ic={} omega={:7.2f} blocked={:6.1%} '
                      'varRatio ideal(A,B)=({:6.3f},{:6.3f}) gated=({:6.3f},{:6.3f}) '
                      'meanRMS/SE(A,B)=({:5.2f},{:5.2f})'.format(beta,h,ic,row['omega'],
                      row['blocked_rate_fraction'],row['A_variance_ratio_ideal'],row['B_variance_ratio_ideal'],
                      row['A_variance_ratio_gated'],row['B_variance_ratio_gated'],
                      row['A_mean_rms_residual_gated']/max(row['A_mean_rms_sampling_SE_gated'],1e-30),
                      row['B_mean_rms_residual_gated']/max(row['B_mean_rms_sampling_SE_gated'],1e-30)))
    path = Path(args.csv); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Saved',path,'Elapsed {:.1f}s'.format(time.perf_counter()-tic))


if __name__ == '__main__':
    main()
