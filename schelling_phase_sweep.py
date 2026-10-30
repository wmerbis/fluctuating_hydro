#!/usr/bin/env python3
"""Resumable 2x21x21 Schelling phase-diagram sweep (finite-volume Gaussian noise).

Install alongside your fhd/ package. Requires fhd/observables.py from the
previous observables package and the user's sigma-regularized fhd/FHD_2D.py.
No changes to the solver/package are made.

Run a tiny test first:
    python schelling_phase_sweep.py --n-grid 3 --nsteps 100 --frames 10 --workers 2 --out sweep_smoke
Full sweep:
    python schelling_phase_sweep.py --workers 4 --out phase_sweep
Only consolidate existing results:
    python schelling_phase_sweep.py --aggregate-only --out phase_sweep
Deterministic time step refinement pilot:
    python schelling_phase_sweep.py --dt-test --dt-test-time 0.5

Output:
  out/{reciprocal,nonreciprocal}/bNN_xNN.npz    one point/replica; resumable
  out/summary.csv, out/{reciprocal,nonreciprocal}_grids.npz
  out/failures/*.txt, out/config.json

Physics and interpretation:
* The phenomenological Cahn-Hilliard stiffness is controlled by sigma,
  while h=dx/sqrt(people_per_cell) controls conservative Gaussian noise.
* Dynamic spectra use a low-k cutoff and a strided saved trajectory to keep
  memory/CPU manageable. You can change these settings on the command line.
* Dissimilarity follows the conventional two-group D index. If your installed
  fhd.operations supplies dissimilarity and mean_relative_entropy, they are
  evaluated as well as independent, explicitly documented entropy measures.
* The white-noise reference is the measured Gaussian initial field before
  evolution (estimated covariance under spatially IID Gaussian sampling).
* No hard-coded phase labels: C is a channel discriminator only when there is
  statistically resolvable excess spectral power.
"""
from __future__ import annotations

import os
# Avoid 4 processes x NBLAS/Numba threads oversubscribing all cores.
for _thread_var in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
):
    os.environ[_thread_var] = '1'

import argparse
import csv
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FAMILIES = ('reciprocal', 'nonreciprocal')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', '--n-cores', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('schelling_phase_sweep'))
    p.add_argument('--n-grid', type=int, default=21, help='Grid points along each axis')
    p.add_argument('--family', choices=('both', *FAMILIES), default='both')
    p.add_argument('--beta-min', type=float, default=0.)
    p.add_argument('--beta-max', type=float, default=3.)
    p.add_argument('--N', nargs=2, type=int, default=(128,128))
    p.add_argument('--L', nargs=2, type=float, default=(50.,50.))
    p.add_argument('--people-per-cell', type=float, default=400.)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('--D', nargs=2, type=float, default=(1.,1.))
    p.add_argument('--dt', type=float, default=1e-3)
    p.add_argument('--nsteps', type=int, default=10000)
    p.add_argument('--frames', type=int, default=500)
    p.add_argument('--rho-a', type=float, default=0.35)
    p.add_argument('--rho-b', type=float, default=0.35)
    p.add_argument('--initial-std', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--replicates', type=int, default=1, help='Independent realizations per parameter point')
    p.add_argument('--noise', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--bc', choices=('Neumann','periodic'), default='Neumann')
    p.add_argument('--bins', type=int, default=40)
    p.add_argument('--k-band', nargs=2, type=float, default=None,
                   help='For excess order parameters (physical k); default all spatial modes')
    p.add_argument('--burn-fraction', type=float, default=0.5,
                   help='Fraction of SAVED frames dropped when calculating late-time means/dynamics')
    p.add_argument('--dynamic-kmax', type=float, default=3.,
                   help='Only retain |k| <= this for spatiotemporal analysis; --dynamic-kmax 0 disables cutoff')
    p.add_argument('--dynamic-stride', type=int, default=4,
                   help='Use every nth frame for temporal spectra/covariances')
    p.add_argument('--max-lag', type=int, default=32,
                   help='Maximum lag in downsampled saved frames')
    p.add_argument('--no-dynamics', action='store_true', help='Disable temporal spectra (faster, less RAM)')
    p.add_argument('--snapshots', type=int, default=5, help='Number of equally spaced saved density snapshots')
    p.add_argument('--save-trajectory', action='store_true',
                   help='Also save entire (2,frames+1,Nx,Ny) trajectory as float32')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--aggregate-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dt-test', action='store_true', help='Run deterministic dt, dt/2, dt/4 pilot and exit')
    p.add_argument('--dt-test-family', choices=FAMILIES, default='reciprocal')
    p.add_argument('--dt-test-x', type=float, default=-1.)
    p.add_argument('--dt-test-beta', type=float, default=3.)
    p.add_argument('--dt-test-time', type=float, default=0.5)
    p.add_argument('--max-jobs', type=int, default=None, help='Limit scheduled jobs for a smoke test')
    return p.parse_args(argv)


def kappa_for(family, x):
    return np.array([[1., x], [x if family == 'reciprocal' else -x, 1.]], dtype=float)


def validate(args):
    if args.workers < 1 or args.n_grid < 1 or args.replicates < 1:
        raise ValueError('workers, n-grid and replicates must be >= 1')
    nx,ny=args.N
    lx,ly=args.L
    if min(nx,ny)<2 or min(lx,ly)<=0 or args.people_per_cell<=0:
        raise ValueError('Invalid spatial grid, domain, or cell capacity')
    if not (0<args.rho_a and 0<args.rho_b and args.rho_a+args.rho_b<1):
        raise ValueError('Species means must be positive and have vacancies')
    if args.dt<=0 or args.nsteps<1 or args.frames<1 or args.frames>args.nsteps:
        raise ValueError('Require dt>0, nsteps>=1, 1<=frames<=nsteps')
    if args.nsteps % args.frames:
        raise ValueError('nsteps must be divisible by frames: otherwise solver output does not have uniform, fully populated time samples')
    if not (0<=args.burn_fraction<1): raise ValueError('burn-fraction must be in [0,1)')
    if args.dynamic_stride<1 or args.max_lag<0 or args.snapshots<1:
        raise ValueError('Invalid dynamic-stride/max-lag/snapshots')
    dx,dy=lx/nx,ly/ny
    h=math.sqrt(dx*dy/args.people_per_cell)
    # Safe sufficient forward-Euler positivity bound for bounded directed hop rate <=2D/delta**2.
    cfl=1/(4*max(args.D)*(dx**-2+dy**-2))
    print(f'Grid={nx}x{ny}, dx={dx:.6g}, dy={dy:.6g}, h={h:.7g}, '
          f'capacity={args.people_per_cell:g}, sigma={args.sigma:g}')
    print(f'dt={args.dt:g}, Dmax*dt/min(dx,dy)^2={max(args.D)*args.dt/min(dx,dy)**2:.5g}, '
          f'bounded-hop positivity dt limit ~{cfl:.5g}; total T={args.nsteps*args.dt:g}')
    if args.dt>cfl:
        print('WARNING: dt exceeds a sufficient bounded-hop forward-Euler positivity bound.',file=sys.stderr)
    print('NOTE: passing this bound does NOT establish convergence of the nonlinear patterns.')
    return h


def make_jobs(args):
    result=[]
    families=FAMILIES if args.family=='both' else (args.family,)
    betas=np.linspace(args.beta_min,args.beta_max,args.n_grid)
    for fam in families:
        xs=np.linspace(-1.,1.,args.n_grid) if fam=='reciprocal' else np.linspace(0.,1.,args.n_grid)
        for bi,beta in enumerate(betas):
            for xi,x in enumerate(xs):
                for rep in range(args.replicates):
                    # Stable across resume/worker changes.
                    id_f=FAMILIES.index(fam)
                    seed=args.seed+10_000_000*id_f+100_000*bi+1000*xi+rep
                    sub=f'{fam}/b{bi:02d}_x{xi:02d}_r{rep:02d}.npz'
                    result.append(dict(family=fam,bi=bi,xi=xi,rep=rep,seed=int(seed),
                                       beta=float(beta),x=float(x),sub=sub))
    return result


def make_sim_and_initial(args, seed):
    import fhd
    N=tuple(args.N)
    L=tuple(args.L)
    sim=fhd.fhd_2d(L,N,bc=args.bc,fft=False,
                   schelling_flux='finite_volume',
                   projection_mode='redistribute',
                   use_numba_projection=True,
                   numba_projection_threads=1,
                   voter_noise_mode='wright_fisher',
                   wf_gaussian_threshold=0.01,
                   reaction_diagnostic=True)
    rng=np.random.default_rng(int(seed))
    phi=np.empty((2,)+N,dtype=np.float64)
    phi[0]=args.rho_a+args.initial_std*rng.standard_normal(N)
    phi[1]=args.rho_b+args.initial_std*rng.standard_normal(N)
    # Ensure independent initial Gaussian IC and stochastic evolution seeds.
    phi=sim.project_density_fast(phi)
    sim.set_seed(int(seed)+7919)
    return sim,phi


def legacy_indices(a):
    """Legacy fhd.operations metrics if present. Else use independent explicit definitions.

    Standard dissimilarity D=0.5 sum_i |A_i/A_tot - B_i/B_tot|.
    Entropic segregation is the occupancy-weighted KL of local A/B
    composition against the global A/B composition, plus entropy normalized
    by its global random-mixing entropy H_global. These two measures are
    separate from *legacy_mean_relative_entropy*, which is used verbatim if
    installed and may have different weighting/conventions.
    """
    try:
        from fhd import operations as ops
        old_D=getattr(ops,'dissimilarity',None)
        old_KL=getattr(ops,'mean_relative_entropy',None)
    except ImportError:
        old_D=old_KL=None
    nt=a.shape[1]
    D=np.empty(nt)
    KL=np.empty(nt)
    H=np.empty(nt)
    Hnorm=np.empty(nt)
    legacy_D=np.full(nt,np.nan)
    legacy_KL=np.full(nt,np.nan)
    for i in range(nt):
        ph=a[:,i]
        A,B=ph[0],ph[1]
        a_tot=float(A.sum()); b_tot=float(B.sum())
        D[i]=0.5*np.abs(A/max(a_tot,1e-300)-B/max(b_tot,1e-300)).sum()
        occ=A+B
        pbar=a_tot/max(a_tot+b_tot,1e-300)
        mask=occ>1e-12
        c=np.divide(A,occ,out=np.zeros_like(A),where=mask)
        # x log(x/y) returns 0 at x=0, conventional KL divergence.
        kl=np.zeros_like(c)
        np.multiply(c,np.log(np.maximum(c,1e-300)/max(pbar,1e-300)),out=kl,where=mask)
        d=1-c
        kl+=d*np.log(np.maximum(d,1e-300)/max(1-pbar,1e-300))
        KL[i]=np.sum(occ[mask]*kl[mask])/max(np.sum(occ[mask]),1e-300)
        ent=-(c*np.log(np.maximum(c,1e-300))+(1-c)*np.log(np.maximum(1-c,1e-300)))
        H[i]=np.sum(occ[mask]*ent[mask])/max(np.sum(occ[mask]),1e-300)
        HG=-(pbar*np.log(max(pbar,1e-300))+(1-pbar)*np.log(max(1-pbar,1e-300)))
        Hnorm[i]=KL[i]/HG if HG>0 else np.nan
        if callable(old_D):
            try: legacy_D[i]=float(old_D(ph))
            except (TypeError,ValueError,ZeroDivisionError): pass
        if callable(old_KL):
            try: legacy_KL[i]=float(old_KL(ph))
            except (TypeError,ValueError,ZeroDivisionError): pass
    return dict(dissimilarity=D,entropy_kl=KL,local_shannon_entropy=H,
                normalized_entropy_segregation=Hnorm,
                legacy_dissimilarity=legacy_D,
                legacy_mean_relative_entropy=legacy_KL)


def make_result(a,args,job,sim,elapsed):
    from fhd.observables import (
        spatial_structure_factor, excess_spectral_power, segregation_index,
        occupation_index, connected_time_lag_covariance,
        spatiotemporal_structure_factor,
    )
    a=np.asarray(a)
    n_frames=a.shape[1]
    dt_saved=args.dt*(args.nsteps//args.frames)
    times=np.arange(n_frames)*dt_saved
    burn=min(n_frames-2,int(args.burn_fraction*(n_frames-1)))
    late=slice(burn,None)
    cap=args.people_per_cell
    spatial=spatial_structure_factor(a,args.L,bc=args.bc,n_bins=args.bins)
    excess=excess_spectral_power(a,args.L,bc=args.bc,n_bins=args.bins,
                                 reference='white',k_band=args.k_band)
    segreg=segregation_index(a,cell_capacity=cap,subtract_mixing=True)
    occupation=occupation_index(a,reference='initial')
    legacy=legacy_indices(a)
    # Spatial means and variances of n,m,vacancy. Global totals must be conserved.
    n=a[0]+a[1]; m=a[0]-a[1]; v=1-n
    series={
      'mean_A':a[0].mean(axis=(-2,-1)),
      'mean_B':a[1].mean(axis=(-2,-1)),
      'mean_n':n.mean(axis=(-2,-1)),
      'mean_m':m.mean(axis=(-2,-1)),
      'mean_vacancy':v.mean(axis=(-2,-1)),
      'total_A':a[0].sum(axis=(-2,-1)),
      'total_B':a[1].sum(axis=(-2,-1)),
      'total_vacancy':v.sum(axis=(-2,-1)),
      'variance_A':a[0].var(axis=(-2,-1)),
      'variance_B':a[1].var(axis=(-2,-1)),
      'variance_n':n.var(axis=(-2,-1)),
      'variance_m':m.var(axis=(-2,-1)),
      'variance_vacancy':v.var(axis=(-2,-1)),
      'std_n':n.std(axis=(-2,-1)),
      'std_m':m.std(axis=(-2,-1)),
      'std_vacancy':v.std(axis=(-2,-1)),
      'I_seg':segreg['I_seg'],
      'I_seg_excess':segreg['excess'],
      'I_seg_mixing_baseline':segreg['mixing_baseline'],
      'I_occ':occupation['I_occ'],
      'I_occ_excess':occupation['excess'],
      'P_plus':excess['P_plus'],
      'P_minus':excess['P_minus'],
      'C':excess['C'],
    }
    series.update(legacy)
    result={
      'family':np.array(job['family']), 'beta':np.float64(job['beta']),
      'x':np.float64(job['x']),'seed':np.int64(job['seed']),
      'replicate':np.int64(job['rep']),
      'kappa':kappa_for(job['family'],job['x']),
      'dt':np.float64(args.dt), 'dt_saved':np.float64(dt_saved),
      'sigma':np.float64(args.sigma),
      'h':np.float64(math.sqrt(np.prod(np.array(args.L)/np.array(args.N))/args.people_per_cell)),
      'people_per_cell':np.float64(cap),
      'N':np.asarray(args.N,dtype=int), 'L':np.asarray(args.L,dtype=float),
      'rho_bar':np.array([args.rho_a,args.rho_b]),
      'nsteps':np.int64(args.nsteps), 'noise':np.bool_(args.noise),
      'min_species_full':np.float64(a.min()),
      'min_vacancy_full':np.float64(v.min()),
      'times':times.astype('float32'), 'burn_frame':np.int64(burn),
      'elapsed_seconds':np.float64(elapsed),
      'k':spatial['k'].astype('float32'),
      'k_edges':spatial['k_edges'].astype('float32'),
      'k_counts':spatial['counts'].astype('int32'),
      'S_dct':spatial['S'].real.astype('float32'),
      'S_plus':spatial['S_plus'].astype('float32'),
      'S_minus':spatial['S_minus'].astype('float32'),
      'S_reference':excess['baseline'].real.astype('float32'),
      'S_excess':excess['excess'].real.astype('float32'),
      'excess_plus':excess['excess_plus'].astype('float32'),
      'excess_minus':excess['excess_minus'].astype('float32'),
      'C_threshold':np.float64(excess['order_threshold']),
    }
    result.update({k:np.asarray(val,dtype=np.float32) for k,val in series.items()})
    snap_ids=np.unique(np.linspace(0,n_frames-1,min(args.snapshots,n_frames),dtype=int))
    result['snapshot_indices']=snap_ids
    result['snapshots']=a[:,snap_ids].astype(np.float32)
    if args.save_trajectory:
        result['phi_run']=a.astype(np.float32)
    if not args.no_dynamics:
        # Dynamic cutoff is deliberately separate from all-scale DCT spectrum.
        dyn=a[:,burn::args.dynamic_stride]
        if dyn.shape[1]>=3:
            kmax=args.dynamic_kmax if args.dynamic_kmax>0 else None
            dtd=dt_saved*args.dynamic_stride
            maxlag=min(args.max_lag,dyn.shape[1]-1)
            lags=np.arange(0,maxlag+1,dtype=int)
            tc=connected_time_lag_covariance(dyn,dtd,args.L,bc=args.bc,
                        n_bins=args.bins,k_max=kmax,lags=lags,burn_in=0)
            st=spatiotemporal_structure_factor(dyn,dtd,args.L,bc=args.bc,
                        n_bins=args.bins,k_max=kmax,burn_in=0,window='hann')
            result.update({
                'dynamic_frame_indices':np.arange(burn,n_frames,args.dynamic_stride,dtype=np.int32),
                'k_dynamic':tc['k'].astype('float32'),
                'tau':tc['tau'].astype('float32'),
                'C_k_tau':tc['C'].astype('complex64'),
                'C_plus_tau':tc['C_plus'].astype('float32'),
                'C_minus_tau':tc['C_minus'].astype('float32'),
                'omega':st['omega'].astype('float32'),
                'S_k_omega':st['S'].astype('complex64'),
                'S_plus_omega':st['S_plus'].astype('float32'),
                'S_minus_omega':st['S_minus'].astype('float32'),
            })
    # Projection/limiter diagnostics: flat numerics only, avoid large mutable history.
    diag=sim._work.get('projection_diag',{}) if hasattr(sim,'_work') else {}
    for name,val in diag.items():
        if isinstance(val,(int,float,np.integer,np.floating)) and not isinstance(val,bool):
            result['diag_'+name]=np.asarray(val)
    # Scalar late-time diagnostics. 'C' is undefined if no reliable excess.
    summary={'family':job['family'],'beta':job['beta'],'x':job['x'],
             'bi':job['bi'],'xi':job['xi'],'rep':job['rep'],
             'seed':job['seed'],'time':args.nsteps*args.dt,
             'elapsed_seconds':elapsed,
             'h':float(result['h']),'sigma':args.sigma,
             'noise':int(args.noise), 'status':'ok'}
    for name,values in series.items():
        tail=np.asarray(values[late],dtype=float)
        finite=tail[np.isfinite(tail)]
        summary[name+'_late']=float(np.mean(finite)) if finite.size else float('nan')
        # Temporal fluctuations of each global scalar observable.
        summary[name+'_temporal_std']=float(np.std(finite)) if finite.size else float('nan')
    summary['C_valid_fraction']=float(np.isfinite(series['C'][late]).mean())
    summary['min_species']=float(a.min())
    summary['min_vacancy']=float(v.min())
    summary['max_mass_drift']=float(max(
        np.max(np.abs(series['total_A']-series['total_A'][0])),
        np.max(np.abs(series['total_B']-series['total_B'][0]))))
    for diag_name in ('n_low_entries','n_simplex_cells',
                      'n_schelling_noise_limited_faces','schelling_noise_transfer_suppressed'):
        summary[diag_name]=float(diag.get(diag_name,0))
    # Characteristic k from the positive late-time excess in each channel.
    counts=np.asarray(spatial['counts'])
    for tag in ('plus','minus'):
        radial=np.nanmean(np.asarray(excess['excess_'+tag][late]),axis=0)
        use=(counts>0)&np.isfinite(radial)
        if args.k_band is not None:
            use&=(spatial['k']>=args.k_band[0])&(spatial['k']<=args.k_band[1])
        result_k=spatial['k'][use]
        result_power=radial[use]
        summary['k_peak_'+tag]=float(result_k[np.argmax(result_power)]) if result_power.size and np.max(result_power)>0 else float('nan')
    return result,summary


def worker(job, cfg):
    # Rebuild argparse-compatible namespace inside spawned worker.
    args=argparse.Namespace(**cfg)
    dest=args.out/job['sub']
    dest.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:
        sim,phi=make_sim_and_initial(args,job['seed'])
        param=dict(D=np.asarray(args.D,dtype=float),
                   kappa=kappa_for(job['family'],job['x']),
                   beta=job['beta'],
                   rho_bar=np.array([args.rho_a,args.rho_b]),
                   h=math.sqrt(np.prod(np.asarray(args.L)/np.asarray(args.N))/args.people_per_cell),
                   sigma=args.sigma,
                   theta_face='midpoint',schelling_noise_mobility='symmetric')
        # This is the user's sigma-enabled version; fail if it silently uses h.
        series=sim.run(phi,param,args.nsteps,args.dt,args.noise,args.frames,
                       scheme='FE',model='Schelling',use_fastpath=False,verbatum=False,
                       diagnostic_interval=None)
        if not hasattr(sim,'_schelling_new'):
            raise RuntimeError('Expected Schelling parameter cache absent; check installed solver')
        # Verify sigma is actually used by comparing with Theta at the mean.
        from fhd.FHD_2D import schelling_theta
        theta_bar=schelling_theta(np.asarray(sim._schelling_new['rho_bar']).reshape(2,1,1), sim._schelling_new['coeff'])[:,:,0,0]
        gamma_expected=2*args.sigma**2*theta_bar
        gamma_actual=np.asarray(sim._schelling_new['Gamma'])
        if not np.allclose(gamma_actual,gamma_expected,rtol=1e-9,atol=1e-10):
            raise RuntimeError('Solver Gamma does not equal 2*sigma**2*Theta(rho_bar)! '
                               f'Got {gamma_actual}, expected {gamma_expected}; use the sigma-patched solver.')
        result,summary=make_result(series,args,job,sim,time.monotonic()-start)
        tmp=dest.with_name(dest.name+'.tmp')
        with tmp.open('wb') as f: np.savez_compressed(f,**result)
        os.replace(tmp,dest)
        return summary
    except BaseException:
        err=args.out/'failures'/job['sub'].replace('/','__')
        err.parent.mkdir(parents=True,exist_ok=True)
        err.with_suffix('.txt').write_text(traceback.format_exc())
        raise


def summarize_existing(args,jobs):
    rows=[]
    for job in jobs:
        path=args.out/job['sub']
        if not path.exists(): continue
        # Read no time-series from npz for consolidation beyond scalar fields.
        try:
            with np.load(path,allow_pickle=False) as f:
                # Extract lightweight scalar diagnostics and calculate late means of series.
                burn=int(f['burn_frame']); row={
                    'family':str(f['family']),'beta':float(f['beta']),
                    'x':float(f['x']),'bi':job['bi'],'xi':job['xi'],
                    'rep':int(f['replicate']),'seed':int(f['seed']),
                    'time':float(f['times'][-1]),'h':float(f['h']),
                    'sigma':float(f['sigma']),
                    'noise':int(f['noise']),
                    'elapsed_seconds':float(f['elapsed_seconds']),
                    'status':'ok'}
                stat=['mean_A','mean_B','mean_n','mean_m','mean_vacancy',
                      'total_A','total_B','total_vacancy',
                      'variance_A','variance_B','variance_n','variance_m','variance_vacancy',
                      'std_n','std_m','std_vacancy','I_seg','I_seg_excess','I_occ',
                      'I_occ_excess','P_plus','P_minus','C',
                      'dissimilarity','entropy_kl','local_shannon_entropy',
                      'normalized_entropy_segregation','legacy_dissimilarity',
                      'legacy_mean_relative_entropy']
                for key in stat:
                    if key not in f: continue
                    tail=np.asarray(f[key][burn:],dtype=float)
                    finite=tail[np.isfinite(tail)]
                    row[key+'_late']=float(finite.mean()) if len(finite) else float('nan')
                    row[key+'_temporal_std']=float(finite.std()) if len(finite) else float('nan')
                row['C_valid_fraction']=float(np.isfinite(f['C'][burn:]).mean())
                row['min_species']=float(f['min_species_full']) if 'min_species_full' in f else float(np.min(f['snapshots']))
                row['min_vacancy']=float(f['min_vacancy_full']) if 'min_vacancy_full' in f else float(np.min(1.-np.sum(f['snapshots'],axis=0)))
                row['max_mass_drift']=float(max(
                    np.max(np.abs(f['total_A']-f['total_A'][0])),
                    np.max(np.abs(f['total_B']-f['total_B'][0]))))
                for name in ('n_low_entries','n_simplex_cells',
                             'n_schelling_noise_limited_faces','schelling_noise_transfer_suppressed'):
                    key='diag_'+name
                    row[name]=float(f[key]) if key in f else 0.
                for tag in ('plus','minus'):
                    radial=np.nanmean(np.asarray(f['excess_'+tag][burn:]),axis=0)
                    valid=(np.asarray(f['k_counts'])>0)&np.isfinite(radial)
                    kval=np.asarray(f['k'])[valid]
                    rval=radial[valid]
                    row['k_peak_'+tag]=float(kval[np.argmax(rval)]) if rval.size and np.max(rval)>0 else float('nan')
                rows.append(row)
        except Exception as e:
            print(f'Warning: could not read {path}: {e}',file=sys.stderr)
    if not rows:
        print('No completed results to consolidate.')
        return
    args.out.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    summary_path=args.out/'summary.csv'
    with summary_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    for family in FAMILIES:
        family_rows=[r for r in rows if r['family']==family]
        if not family_rows: continue
        keys=[c for c in columns if c.endswith('_late') or c in (
            'min_species','min_vacancy','max_mass_drift','C_valid_fraction',
            'k_peak_plus','k_peak_minus','n_schelling_noise_limited_faces')]
        matrices={}
        for key in keys:
            grid=np.full((args.n_grid,args.n_grid,args.replicates),np.nan)
            for row in family_rows:
                if key in row:
                    grid[row['bi'],row['xi'],row['rep']]=row[key]
            matrices[key]=np.nanmean(grid,axis=-1) if args.replicates>1 else grid[:,:,0]
        xs=np.linspace(-1,1,args.n_grid) if family=='reciprocal' else np.linspace(0,1,args.n_grid)
        np.savez_compressed(args.out/f'{family}_grids.npz',
                            beta=np.linspace(args.beta_min,args.beta_max,args.n_grid),
                            x=xs,**matrices)
    print(f'Consolidated {len(rows)} runs -> {summary_path}')


def dt_pilot(args):
    print('\nDeterministic dt convergence pilot (same initial condition; noise disabled).')
    family=args.dt_test_family
    kappa=kappa_for(family,args.dt_test_x)
    base=math.sqrt(np.prod(np.asarray(args.L)/np.asarray(args.N))/args.people_per_cell)
    vals=[]
    for scale in (1,2,4):
        dt=args.dt/scale
        nsteps=round(args.dt_test_time/dt)
        if nsteps<1: raise ValueError('dt-test-time too short')
        sim,phi=make_sim_and_initial(args,args.seed)
        param=dict(D=np.asarray(args.D),kappa=kappa,beta=args.dt_test_beta,
                   rho_bar=np.array([args.rho_a,args.rho_b]),
                   h=base,sigma=args.sigma,theta_face='midpoint',
                   schelling_noise_mobility='symmetric')
        start=time.monotonic()
        output=sim.run(phi,param,nsteps,dt,False,1,scheme='FE',model='Schelling',
                       use_fastpath=False,verbatum=False,diagnostic_interval=None)
        vals.append(output[:,-1].copy())
        print(f' dt={dt:.4g}, steps={nsteps}, t={nsteps*dt:.4g}, '
              f'elapsed={time.monotonic()-start:.2f}s, min={vals[-1].min():.3g}')
    den=max(float(np.linalg.norm(vals[2])),1e-15)
    err12=float(np.linalg.norm(vals[0]-vals[1])/den)
    err24=float(np.linalg.norm(vals[1]-vals[2])/den)
    print(f'Relative final-state RMS dt versus dt/2: {err12:.5g}; '
          f'dt/2 versus dt/4: {err24:.5g}')
    print('For unstable nonlinear patterns, RMS can grow from tiny timing errors. '
          'Also compare order parameters and spectra at the same physical times.')


def main(argv=None):
    args=parse_args(argv)
    args.out=args.out.resolve()
    h=validate(args)
    if args.dt_test:
        dt_pilot(args); return 0
    jobs=make_jobs(args)
    if args.aggregate_only:
        summarize_existing(args,jobs); return 0
    pending=[j for j in jobs if args.overwrite or not (args.out/j['sub']).exists()]
    if args.max_jobs is not None: pending=pending[:args.max_jobs]
    if args.dry_run:
        print(f'Would schedule {len(pending)} of {len(jobs)} jobs using {args.workers} workers.')
        print('First jobs:',pending[:3]); return 0
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=vars(args).copy()
    # Fail fast on incompatible resumed runs: filenames depend on indices, not physics.
    config_path=args.out/'config.json'
    config_data={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()}
    physical_keys=('n_grid','beta_min','beta_max','N','L','people_per_cell','sigma',
                   'D','dt','nsteps','frames','rho_a','rho_b','initial_std',
                   'seed','replicates','noise','bc','bins','k_band',
                   'burn_fraction','dynamic_kmax','dynamic_stride','max_lag',
                   'no_dynamics','snapshots','save_trajectory')
    if config_path.exists() and not args.overwrite:
        prev=json.loads(config_path.read_text())
        changed=[k for k in physical_keys if k in prev and prev[k]!=config_data[k]]
        if changed:
            if not args.overwrite:
                raise ValueError('Existing output directory was generated with different '
                                 f'parameters {changed}. Use a new --out directory '
                                 'or --overwrite for an intentional replacement.')
            # Invalidate ALL old points, even if --max-jobs limits the new run.
            for fam in FAMILIES:
                for old in (args.out/fam).glob('*.npz'):
                    old.unlink()
            for old in args.out.glob('*_grids.npz'):
                old.unlink()
            print(f'Configuration changed ({changed}); deleted incompatible old NPZ outputs.')
    config_path.write_text(json.dumps(config_data,indent=2))
    print(f'Scheduled {len(pending)} / {len(jobs)} runs on {args.workers} worker(s). '
          'Each saves its own NPZ; restarting skips existing results.')
    errors=0
    if pending:
        # spawn avoids inheriting Numba/OpenMP state from the parent.
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
            futures={executor.submit(worker,j,cfg):j for j in pending}
            for done,future in enumerate(as_completed(futures),1):
                job=futures[future]
                try:
                    s=future.result()
                    print(f'[{done}/{len(pending)}] {job["family"]} '
                          f'beta={job["beta"]:.3f} x={job["x"]:.3f} '
                          f'late C={s["C_late"]:+.3f} '
                          f'Iseg={s["I_seg_late"]:.4g} '
                          f'Iocc={s["I_occ_late"]:.4g} '
                          f'elapsed={s["elapsed_seconds"]:.1f}s',flush=True)
                except Exception as exc:
                    errors+=1
                    print(f'[{done}/{len(pending)}] FAILED {job["sub"]}: {exc}',file=sys.stderr,flush=True)
    summarize_existing(args,jobs)
    if errors:
        print(f'{errors} failed jobs; see {args.out}/failures',file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
