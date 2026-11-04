#!/usr/bin/env python3
"""Plot Schelling phase sweeps, with finite-domain linear-stability overlays.

The heatmaps come from schelling_phase_sweep.py, whose grids have shape
(n_beta, n_x).  Plots here have beta on the HORIZONTAL axis and x VERTICAL.

The exact 2+3-spectator Theta is evaluated at the homogeneous background.
For the sigma-regularized model Gamma = 2*sigma**2 * Theta_bar,

    L(q) = -q**2 [ J + q**2 B ],
    B_ab = 2*sigma**2 * D_a * rho_a * rho_0 * Theta_ab.

For periodic and cell-centered Neumann boundaries, q**2 are the eigenvalues
of the actual second-order finite-volume negative Laplacian.  A real 2x2
matrix M has eigenvalues of positive real part iff trace(M)>0 and det(M)>0.
The finite-grid threshold below checks this condition over *all* nonzero
Neumann/periodic modes, without building a huge k-by-parameter tensor.
For reciprocal parameters, both exchange-symmetric branches lambda_+ and
lambda_- are plotted independently, EVEN AFTER the other branch is unstable.
For nonreciprocal parameters, the onset of complex eigenvalues is indicated
by the minimum discriminant Delta(q) crossing zero over allowed modes.

Usage:
    python plot_schelling_phase_diagram.py --out phase_sweep --save phase_diagrams.png
    python plot_schelling_phase_diagram.py --out phase_sweep --longwave
    python plot_schelling_phase_diagram.py --out phase_sweep --no-theory
    python plot_schelling_phase_diagram.py --out phase_sweep --keys Omega_max r_max

Overrides (otherwise values are read from out/config.json):
    --sigma 0.5 --N 128 128 --L 50 50 --bc Neumann
    --rho-a 0.35 --rho-b 0.35 --D 1 1 --theory-grid 201
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.colors import CenteredNorm

from fhd.observables import (
    estimate_linear_eigenvalues,
    eigenvalues_from_covariance,
)
 

FAMILIES = ('reciprocal', 'nonreciprocal')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('schelling_phase_sweep'))
    parser.add_argument('--keys', nargs='+',
                        default=['C_late', 'entropy_kl_late', 'I_occ_late', 'oscillation_peak'])
    parser.add_argument('--save', type=Path, default=None)
    parser.add_argument('--no-theory', action='store_true',
                        help='Do not overlay the stability threshold')
    parser.add_argument('--longwave', action='store_true',
                        help='Also show infinite-system J-only threshold (gray dash-dot); '
                             'for nonreciprocal interactions, show the J discriminant-zero curve')
    parser.add_argument('--theory-grid', type=int, default=201,
                        help='Resolution in both beta and x for theoretical contours')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override sigma from sweep config')
    parser.add_argument('--N', nargs=2, type=int, default=None)
    parser.add_argument('--L', nargs=2, type=float, default=None)
    parser.add_argument('--bc', choices=['Neumann','periodic'], default=None)
    parser.add_argument('--rho-a', type=float, default=None)
    parser.add_argument('--rho-b', type=float, default=None)
    parser.add_argument('--D', nargs=2, type=float, default=None)
    parser.add_argument('--osc-k-band', nargs=2, type=float, default=None,
                        help='Physical wavenumber band for temporal diagnostics; default all saved dynamic k bins')
    parser.add_argument('--osc-min-peak-ratio', type=float, default=1.10,
                        help='Require nonzero-frequency spectral peak > this multiple of low-frequency power')
    parser.add_argument('--osc-max-lag-fraction', type=float, default=0.75,
                        help='Use lags up to this fraction of the post-burn-in trajectory duration for diagnostics (default 0.75); all C(k,tau) values remain saved')
    parser.add_argument('--eig-k-range', nargs=2, type=float, default=(0.5,2.5),
                        metavar=('KMIN','KMAX'),
                        help='Physical k interval scanned for eigenvalue maxima (default 0.5 to 2.5)')
    parser.add_argument('--eig-widths', nargs='+', type=float, default=(0.35,0.5,0.75),
                        help='Physical widths of sliding k windows')
    parser.add_argument('--eig-step', type=float, default=0.1,
                        help='Physical k interval between sliding windows')
    parser.add_argument('--eig-bin-width', type=float, default=0.1,
                        help='Physical width of mode bins for raw phi runs')
    parser.add_argument('--eig-lag', type=int, default=1,
                        help='Lag in saved frames used for empirical propagator')
    parser.add_argument('--eig-min-modes', type=int, default=20,
                        help='Minimum spatial modes per band')
    parser.add_argument('--eig-max-condition', type=float, default=1.e6,
                        help='Maximum covariance condition number')
    parser.add_argument('--eig-min-modulus', type=float, default=0.25,
                        help='Reject weakly predictive discrete eigenvalues below this modulus')
    parser.add_argument('--eig-min-cycles', type=float, default=0.5,
                        help='Require at least this many oscillation cycles during the saved run')
    return parser.parse_args(argv)


def physical_config(args):
    config_path = args.out / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    def get(cli_value, name, default):
        return cli_value if cli_value is not None else config.get(name, default)
    params = dict(
        sigma=float(get(args.sigma, 'sigma', 0.5)),
        N=tuple(map(int, get(args.N, 'N', (128,128)))),
        L=tuple(map(float, get(args.L, 'L', (50.,50.)))),
        bc=get(args.bc, 'bc', 'Neumann'),
        rho=(float(get(args.rho_a, 'rho_a', 0.35)),
             float(get(args.rho_b, 'rho_b', 0.35))),
        D=tuple(map(float, get(args.D, 'D', (1.,1.)))),
    )
    if min(*params['N'], *params['L']) <= 0 or params['sigma'] < 0:
        raise ValueError('N and L must be positive, sigma >= 0')
    if min(*params['rho'], 1.0-sum(params['rho'])) <= 0:
        raise ValueError('Homogeneous density must be in the interior of simplex')
    if min(params['D']) <= 0:
        raise ValueError('D_a must be positive')
    return params


def kappa_family(family, x):
    """Return kappa_ab with array-valued x (any shape)."""
    x = np.asarray(x, dtype=float)
    k = np.empty(x.shape+(2,2), dtype=float)
    k[...,0,0] = k[...,1,1] = 1.
    k[...,0,1] = x
    k[...,1,0] = x if family == 'reciprocal' else -x
    return k


def theta_grid(family, betas, xs, rho):
    """Exact five-spectator Theta; vectorized across beta and x.

    Returns [x_index, beta_index, a, b].  This uses precisely the
    polynomial finite-state average from the supplied derivation.
    """
    ra, rb = rho
    r0 = 1.0 - ra - rb
    betas = np.asarray(betas, dtype=float)
    xs = np.asarray(xs, dtype=float)
    k = kappa_family(family, xs)
    beta_half = 0.5 * betas[None,:]
    theta = np.zeros((len(xs), len(betas), 2, 2), dtype=float)
    for ma in range(3):
        for mb in range(3-ma):
            w_m = comb(2,ma)*comb(2-ma,mb)
            for na in range(4):
                for nb in range(4-na):
                    w_n = comb(3,na)*comb(3-na,nb)
                    w = (w_m*w_n * ra**(ma+na) * rb**(mb+nb)
                         * r0**(5-ma-mb-na-nb))
                    if not w: continue
                    da, db = ma-na, mb-nb
                    for a in range(2):
                        u = da*k[:,a,0] + db*k[:,a,1]
                        base = np.tanh(beta_half * u[:,None])
                        for b in range(2):
                            theta[:,:,a,b] += w * (
                                np.tanh(beta_half*(u+k[:,a,b])[:,None])-base)
    return theta


def linear_matrices_grid(family, betas, xs, rho, D, sigma):
    """J and B in d(delta rho)/dt = -q^2 [J+q^2 B] delta rho."""
    theta=theta_grid(family,betas,xs,rho)
    ra,rb = rho; r0=1-ra-rb
    d=np.asarray(D, dtype=float)
    # The ideal-gas/vacancy exclusion part: J_ab = D_a*(r0*delta_ab+rho_a).
    ideal=np.array([[d[0]*(r0+ra),d[0]*ra],
                    [d[1]*rb,d[1]*(r0+rb)]])
    w = d * np.asarray(rho) * r0
    J=ideal[None,None,:,:] - 10.*w[None,None,:,None]*theta
    B=2.*sigma**2*w[None,None,:,None]*theta
    return J, B


def laplacian_wavenumbers(N,L,bc):
    """Sorted nonzero eigenvalues q^2 of the discrete FV Neumann/periodic -Delta."""
    nx,ny=map(int,N)
    dx,dy=float(L[0])/nx,float(L[1])/ny
    if bc == 'Neumann':
        kx=4./dx**2*np.sin(np.pi*np.arange(nx)/(2*nx))**2
        ky=4./dy**2*np.sin(np.pi*np.arange(ny)/(2*ny))**2
    elif bc == 'periodic':
        kx=4./dx**2*np.sin(np.pi*np.arange(nx)/nx)**2
        ky=4./dy**2*np.sin(np.pi*np.arange(ny)/ny)**2
    else:
        raise ValueError('Only Neumann/periodic supported')
    q2=(kx[:,None]+ky[None,:]).ravel()
    q2=q2[q2>1e-13]
    return np.unique(q2)


def hurwitz_margins(J,B,q2_values):
    """For ALL discrete nonzero q, find min trace(J+q²B), min det(J+q²B).

    The trace is affine in q², and determinant is quadratic.  Consequently
    the exact discrete minima are at endpoints or the available q² nearest
    the quadratic vertex.  No approximate wavenumber sampling is used.
    """
    q2 = np.asarray(q2_values, dtype=float)
    lo,hi=q2[0],q2[-1]
    t0=J[...,0,0]+J[...,1,1]
    t1=B[...,0,0]+B[...,1,1]
    trace_min=np.minimum(t0+lo*t1,t0+hi*t1)

    a0=J[...,0,0]; a1=J[...,0,1]; a2=J[...,1,0]; a3=J[...,1,1]
    b0=B[...,0,0]; b1=B[...,0,1]; b2=B[...,1,0]; b3=B[...,1,1]
    d0=a0*a3-a1*a2
    d1=a0*b3+b0*a3-a1*b2-b1*a2
    d2=b0*b3-b1*b2
    def det_at(q): return d0+q*(d1+q*d2)
    det_min=np.minimum(det_at(lo),det_at(hi))
    # A convex det(q) can have an interior minimum, near -d1/(2*d2).
    qstar=np.full_like(d0,lo)
    np.divide(-d1,2*d2,out=qstar,where=(d2>0))
    qstar=np.where(d2>0,np.clip(qstar,lo,hi),lo)
    indices=np.searchsorted(q2,qstar,side='left')
    q_upper=q2[np.clip(indices,0,len(q2)-1)]
    q_lower=q2[np.clip(indices-1,0,len(q2)-1)]
    det_min=np.minimum(det_min,np.minimum(det_at(q_lower),det_at(q_upper)))
    return trace_min,det_min



def reciprocal_eigenvalue_fields(J, B, q2):
    """Continuation of both exchange-symmetric eigenvalue branches.

    At every nonzero, allowed q, the two eigenvalues of M(q)=J+q² B
    are lambda_+(q)=M_AA+M_AB and lambda_-(q)=M_AA-M_AB.

    For phase-boundary plots we use the *minimum* of each eigenvalue
    over ALL allowed q, not just whichever eigenmode destabilizes first.
    The minimum is found exactly at an endpoint of the allowed q² set,
    because each eigenvalue is affine in q². This is why both zero
    contours continue even inside regions where the other mode is unstable.
    """
    lo, hi = float(q2[0]), float(q2[-1])
    j_plus, j_minus = J[..., 0, 0]+J[..., 0, 1], J[..., 0, 0]-J[..., 0, 1]
    b_plus, b_minus = B[..., 0, 0]+B[..., 0, 1], B[..., 0, 0]-B[..., 0, 1]
    return (np.minimum(j_plus+lo*b_plus, j_plus+hi*b_plus),
            np.minimum(j_minus+lo*b_minus, j_minus+hi*b_minus))


def complex_mode_discriminant(J, B, q2):
    """Minimum discriminant of M(q)=J+q² B across permitted q².

    Delta(q)=(M_AA-M_BB)^2 + 4*M_AB*M_BA, which is quadratic in q².
    Delta < 0 signifies a complex-conjugate pair at that wavelength.
    Its discrete minimum lies at an endpoint or one of the two allowed
    q² values bracketing the quadratic vertex.

    This tests *onset of complex eigenvalues anywhere in the spectrum*,
    not necessarily the leading/growing eigenmode.
    """
    lo, hi = float(q2[0]), float(q2[-1])
    da=J[..., 0, 0]-J[..., 1, 1]
    db=B[..., 0, 0]-B[..., 1, 1]
    c0=da**2 + 4.*J[..., 0, 1]*J[..., 1, 0]
    c1=2.*da*db + 4.*(J[..., 0, 1]*B[..., 1, 0] +
                        B[..., 0, 1]*J[..., 1, 0])
    c2=db**2 + 4.*B[..., 0, 1]*B[..., 1, 0]

    def at(q):
        return c0 + q*(c1+q*c2)

    result=np.minimum(at(lo),at(hi))
    # Only a convex quadratic can have its minimum at the vertex.
    vertex=np.full_like(c0, lo)
    np.divide(-c1,2.*c2,out=vertex,where=c2>0)
    vertex=np.where(c2>0,np.clip(vertex,lo,hi),lo)
    idx=np.searchsorted(q2,vertex)
    upper=q2[np.clip(idx,0,len(q2)-1)]
    lower=q2[np.clip(idx-1,0,len(q2)-1)]
    return np.minimum(result,np.minimum(at(lower),at(upper)))

def theory_data(family, betas, xs, params, longwave=False):
    J,B=linear_matrices_grid(family,betas,xs,params['rho'],params['D'],params['sigma'])
    if longwave:
        # Infinite-system onset q -> 0: q² goes to zero, so sigma drops out.
        q2=np.array([0.])
    else:
        q2=laplacian_wavenumbers(params['N'],params['L'],params['bc'])
    tmin,dmin=hurwitz_margins(J,B,q2)
    # Real 2x2 M has both eigenvalues with positive real parts iff
    # trace(M)>0 AND det(M)>0.  The boundary of this intersection is zero.
    # A unitless rescaling of det avoids obscuring the trace branch.
    # Rescaling is strictly positive and cannot shift the zero contours.
    scale=1. + np.abs(np.trace(J,axis1=-2,axis2=-1))
    margin=np.minimum(tmin,dmin/scale)
    fields=dict(trace_min=tmin,det_min=dmin,margin=margin)
    if family == 'reciprocal':
        # +/- are genuine eigenmodes only when the homogeneous background
        # and the diffusion constants respect A <-> B symmetry.
        if not (np.isclose(params['rho'][0],params['rho'][1]) and
                np.isclose(params['D'][0],params['D'][1])):
            raise ValueError('Reciprocal +/- eigenvalue contours require '
                             'rho_A=rho_B and D_A=D_B')
        fields['lambda_plus'],fields['lambda_minus'] = reciprocal_eigenvalue_fields(J,B,q2)
    else:
        fields['discriminant_min'] = complex_mode_discriminant(J,B,q2)
    return fields


def draw_contour(ax,beta,x,field,*,color,linestyle,linewidth,label=None,zorder=5):
    finite=np.isfinite(field)
    if finite.sum()<4 or not(np.nanmin(field)<0<np.nanmax(field)):
        return False
    # Masked regions are used only for distinguishing the instability type.
    cs=ax.contour(beta,x,np.ma.masked_invalid(field),levels=[0],colors=[color],
                  linestyles=[linestyle],linewidths=[linewidth],zorder=zorder)
    return bool(cs.allsegs and cs.allsegs[0])


def overlay_threshold(ax,family,betas,xs,params,*,fields=None,longwave_fields=None):
    if fields is None:
        fields=theory_data(family,betas,xs,params)
    out=[]
    if family == 'reciprocal':
        # IMPORTANT: do not mask either branch by stability of the other!
        # This continues both lambda_+ = 0 and lambda_- = 0 deep into
        # regions where one of the modes has already become unstable.
        if draw_contour(ax,betas,xs,fields['lambda_plus'],
                        color='black',linestyle='dashed',linewidth=2.2):
            out.append('lambda_plus')
        if draw_contour(ax,betas,xs,fields['lambda_minus'],
                        color='black',linestyle='solid',linewidth=2.2):
            out.append('lambda_minus')
    else:
        # Nonreciprocal family: continue both stability contours
        # throughout the phase diagram, including unstable regions.

        # Stationary branch: det(J + k²B) = 0
        if draw_contour(ax, betas, xs, fields['det_min'],
                        color='black', linestyle='solid', linewidth=2.0):
            out.append('stationary')

        # Hopf-like branch: tr(J + k²B) = 0
        if draw_contour(ax, betas, xs, fields['trace_min'],
                        color='black', linestyle='dashed', linewidth=2.0):
            out.append('oscillatory')

        # Onset of complex eigenvalues
        if draw_contour(ax, betas, xs, fields['discriminant_min'],
                        color='#7a2778', linestyle='dotted',
                        linewidth=2.0, zorder=7):
            out.append('complex_onset')

    if longwave_fields is not None:
        lw=longwave_fields
        # The original infinite-domain stability envelope, for reference.
        if draw_contour(ax,betas,xs,lw['margin'],
                        color='#555555',linestyle='dashdot',linewidth=1.4,zorder=6):
            out.append('longwave')
        if family == 'nonreciprocal':
            # The exact discriminant of J from the user's original plot.
            if draw_contour(ax,betas,xs,lw['discriminant_min'],
                            color='#1a7c93',linestyle='dotted',linewidth=1.5,zorder=8):
                out.append('complex_J_onset')
    return out,fields


# These maps can be obtained from individual existing sweep NPZ files.
# They were NOT included in *_grids.npz by the original sweep script.
EIGEN_KEYS = ('Omega_max','r_max','r_at_Omega_max','k_Omega_max','k_r_max',
              'r_complex_max','Omega_at_r_complex_max','k_r_complex_max')
DYNAMIC_KEYS = ('oscillation_peak', 'oscillation_frequency',
                'oscillation_negative_lobe', 'temporal_antisymmetry') + EIGEN_KEYS


def dynamic_indicators(data, k_band=None, min_peak_ratio=1.10,
                       max_lag_fraction=0.75):
    """Summaries of already-saved DCT/FFT temporal observables.

    oscillation_peak: finite-frequency peak prominence of the spatially
    ring-averaged TRACE S_AA+S_BB, restricted to the given k band. It is
    zero if no resolved nonzero-frequency peak rises above the low-frequency
    background by min_peak_ratio. A 3-point frequency smoothing suppresses
    isolated periodogram spikes. This is an empirical, conservative
    oscillation indicator, NOT a direct test for complex eigenvalues.

    oscillation_negative_lobe: largest normalized negative excursion of
    trace C(k,tau); may also be affected by finite-length time demeaning.

    temporal_antisymmetry: RMS of C_AB(k,tau)-C_BA(k,tau) at tau>0 divided
    by trace C(k,0). Measures broken time-reversal symmetry in cross-lags,
    not uniquely an oscillatory phase.
    """
    result = {key: np.nan for key in DYNAMIC_KEYS}
    if not 0 < max_lag_fraction <= 1:
        raise ValueError('max_lag_fraction must lie in (0,1]')
    if not all(key in data for key in ('k_dynamic', 'omega', 'S_k_omega', 'tau', 'C_k_tau')):
        return result
    k=np.asarray(data['k_dynamic'], dtype=float)
    omega=np.asarray(data['omega'], dtype=float)
    if k.ndim!=1 or omega.ndim!=1 or k.size<1 or omega.size<7:
        return result
    keep=np.isfinite(k)
    if k_band is not None:
        lo,hi=map(float,k_band)
        keep &= (k>=lo)&(k<=hi)
    if not np.any(keep):
        return result

    S=np.asarray(data['S_k_omega'])
    C=np.asarray(data['C_k_tau'])
    tau=np.asarray(data['tau'])
    if S.shape!=(k.size,omega.size,2,2) or C.shape[1:]!=(k.size,2,2):
        return result

    # Trace is invariant under a change from (A,B) to (+,-) basis.
    power=np.real(S[...,0,0]+S[...,1,1])
    with np.errstate(invalid='ignore'):
        P=np.nanmean(power[keep],axis=0)
    zero=int(np.argmin(np.abs(omega)))
    positive=np.flatnonzero(omega>0.)
    if not np.isfinite(P[zero]) or len(positive)<4:
        return result

    # Smooth *within* strictly positive frequencies: avoids an artificial
    # nonzero-frequency peak from the zero-frequency window sidelobe.
    positive_power=P[positive]
    smooth=(positive_power[:-2]+2*positive_power[1:-1]+
            positive_power[2:])*0.25
    # Exclude edge bins and require a genuine local maximum (not a
    # monotonically decaying relaxational spectrum).
    ix_candidates=np.flatnonzero((smooth[1:-1]>smooth[:-2]) &
                                (smooth[1:-1]>=smooth[2:]))+1
    ix_candidates=ix_candidates[np.isfinite(smooth[ix_candidates])]
    if ix_candidates.size:
        im=ix_candidates[np.argmax(smooth[ix_candidates])]
        peak=float(smooth[im]); low=float(max(P[zero],positive_power[0],0.))
        if np.isfinite(peak) and peak>0 and peak>min_peak_ratio*low:
            result['oscillation_peak']=(peak-low)/(peak+low)
            result['oscillation_frequency']=float(omega[positive[im+1]])
        else:
            result['oscillation_peak']=0.
    else:
        result['oscillation_peak']=0.

    # The lag statistic is intentionally a separate diagnostic: time
    # demeaning over a short window can give spurious negative lobes.
    # Full C(tau) may be saved, but extreme lags have very few pairs. Limit
    # the phase score to a chosen fraction of the available record length.
    if len(tau)>1 and C.shape[0]==len(tau):
        n_frames = len(omega)  # one FFT frequency for each post-burn-in frame
        domega = float(np.median(np.diff(omega)))
        if np.isfinite(domega) and domega > 0 and n_frames > 1:
            dt_saved = 2*np.pi/(n_frames*domega)
            lag_ok = (tau > 0) & (tau <= max_lag_fraction*(n_frames-1)*dt_saved + 1.e-12)
        else:
            lag_ok = tau > 0
        trace=np.real(C[...,0,0]+C[...,1,1])
        if np.any(lag_ok):
            with np.errstate(invalid='ignore'):
                trace0=float(np.nanmean(trace[0,keep]))
                lag_trace=np.nanmean(trace[lag_ok][:,keep],axis=1)
            if np.isfinite(trace0) and trace0>0 and np.any(np.isfinite(lag_trace)):
                result['oscillation_negative_lobe']=float(np.clip(-np.nanmin(lag_trace)/trace0,0.,1.))
                cross=C[lag_ok][:,keep,0,1]-C[lag_ok][:,keep,1,0]
                result['temporal_antisymmetry']=float(np.sqrt(np.nanmean(np.abs(cross)**2))/trace0)
    return result

def temporal_diagnostic_summary(data, k_band=None):
    """Inspect why an archived run does or does not have oscillation peaks.

    Accepts a loaded NPZ or mapping, for use interactively from notebooks.
    Retained time span, spectral resolution, and trace-power comparisons
    can distinguish a too-short run from a genuinely non-peaked spectrum.
    """
    k=np.asarray(data['k_dynamic'],float)
    omega=np.asarray(data['omega'],float)
    tau=np.asarray(data['tau'],float)
    S=np.asarray(data['S_k_omega'])
    keep=np.isfinite(k)
    if k_band is not None:
        keep &= (k>=k_band[0]) & (k<=k_band[1])
    if not np.any(keep):
        raise ValueError('No saved spatial bins in k_band')
    P=np.nanmean(np.real(S[keep,:,0,0]+S[keep,:,1,1]),axis=0)
    zero=np.argmin(np.abs(omega))
    positive=np.flatnonzero(omega>0)
    domega=float(np.median(np.diff(omega)))
    duration=2*np.pi/domega if domega>0 else np.nan
    low=float(max(P[zero],P[positive[0]],0)) if len(positive) else np.nan
    finite_positive=positive[np.isfinite(P[positive])]
    peak_i=int(finite_positive[np.argmax(P[finite_positive])]) if len(finite_positive) else None
    peak_power=float(P[peak_i]) if peak_i is not None else np.nan
    return dict(n_k_bins=int(keep.sum()), n_time_frames=len(omega),
                n_saved_lags=len(tau), max_stored_lag=float(tau[-1]),
                post_burn_duration=duration,
                frequency_resolution=domega,
                low_frequency_power=low,
                largest_positive_frequency_power=peak_power,
                largest_positive_to_low_ratio=float(peak_power/low)
                    if peak_i is not None and low>0 else np.nan,
                omega_at_largest_power=float(omega[peak_i])
                    if peak_i is not None else np.nan)


def _mode_counts_for_saved_bins(k, params):
    """Reconstruct the number of modes per saved radial bin using PHYSICAL L."""
    k = np.asarray(k, dtype=float)
    if k.size < 2 or not np.all(np.diff(k) > 0):
        return None
    edges = np.r_[k[0]-(k[1]-k[0])/2,
                  (k[:-1]+k[1:])/2, k[-1]+(k[-1]-k[-2])/2]
    nx, ny = params['N']
    lx, ly = params['L']
    if params['bc'].lower() == 'neumann':
        kx = np.pi*np.arange(nx)/lx
        ky = np.pi*np.arange(ny)/ly
    else:
        kx = 2*np.pi*np.fft.fftfreq(nx, d=lx/nx)
        ky = 2*np.pi*np.fft.fftfreq(ny, d=ly/ny)
    kval = np.hypot(kx[:,None],ky[None,:]).ravel()
    return np.histogram(kval[kval>0], bins=edges)[0]


def eigen_indicators(data, params, *, k_range=(0.5,2.5),
                     widths=(0.35,0.5,0.75), step=0.1,
                     bin_width=0.1, lag_frames=1, min_modes=20,
                     max_condition=1.e6, min_modulus=0.25,
                     min_cycles=0.5):
    """Estimate rates in saved time units from raw phi or archived C_k_tau.

    Prefer raw phi arrays, because archived C_k_tau was ordinarily time
    demeaned by the sweep. Correlation-based results are provisional if
    the simulation was not in a stationary homogeneous linear regime.
    """
    raw_key = next((key for key in ('phi_runs','phi_run') if key in data), None)
    options = dict(k_range=k_range, widths=widths, step=step,
                   min_modes=min_modes, max_condition=max_condition,
                   min_modulus=min_modulus, min_cycles=min_cycles)
    saved={key: float(data[key]) if key in data else np.nan
           for key in EIGEN_KEYS}
    if raw_key is None and all(key in data for key in EIGEN_KEYS):
        # All direct estimator outputs were persisted in the sweep.
        return saved
    if raw_key is not None:
        if 'dt_saved' in data:
            dt = float(np.asarray(data['dt_saved']))
        elif 'tau' in data:
            time = np.asarray(data['tau'],float)
            if len(time)<2:
                return {key:np.nan for key in EIGEN_KEYS}
            dt = float(time[1]-time[0])
        else:
            return {key:np.nan for key in EIGEN_KEYS}
        result = estimate_linear_eigenvalues(
            data[raw_key], dt, L=params['L'], bc=params['bc'],
            bin_width=bin_width, lag_frames=lag_frames, **options)
    elif all(key in data for key in ('C_k_tau','k_dynamic','tau')):
        k=np.asarray(data['k_dynamic'],dtype=float)
        t=np.asarray(data['tau'],dtype=float)
        C=np.asarray(data['C_k_tau'])
        if lag_frames>=len(t):
            return {key:np.nan for key in EIGEN_KEYS}
        counts = _mode_counts_for_saved_bins(k, params)
        # The spectrum has one frequency per saved snapshot; use its length
        # for the period-resolution check, not the maximum requested lag.
        duration = ((len(data['omega'])-1)*(t[1]-t[0])
                    if 'omega' in data and len(t)>1 else float(t[-1]))
        result = eigenvalues_from_covariance(
            k,t,C,counts=counts,lag_index=lag_frames,
            duration=duration if duration>0 else None,**options)
    else:
        return saved
    # Prefer directly saved scalars when available (they may be less biased
    # than archived, time-demeaned covariances); fill missing new diagnostics.
    return {key: saved[key] if np.isfinite(saved[key]) else result[key]
            for key in EIGEN_KEYS}



def dynamic_grid(out, family, beta, xs, key, *, k_band=None, min_peak_ratio=1.10,
                 max_lag_fraction=0.75, eigen_options=None, params=None, 
                 eigen_cache=None):    
    """Build a [beta,x] map from saved per-run NPZs; average replicates."""
    shape=(len(beta),len(xs))
    sums=np.zeros(shape,dtype=float)
    counts=np.zeros(shape,dtype=int)
    folder=out/family
    matched=0
    warned_covariance=False
    for path in folder.glob('b*_x*_r*.npz'):
        match=re.fullmatch(r'b(\d+)_x(\d+)_r(\d+)\.npz',path.name)
        if match is None: continue
        ib,ix,_=map(int,match.groups())
        if ib>=len(beta) or ix>=len(xs): continue
        try:
            with np.load(path,allow_pickle=False) as f:
                # Keep indexing robust if the user copied a file between sweeps.
                if not (np.isclose(float(f['beta']),beta[ib]) and
                        np.isclose(float(f['x']),xs[ix])):
                    continue
                if key in EIGEN_KEYS:
                    if (not warned_covariance and key not in f and
                        not any(k in f for k in ('phi_run','phi_runs')) and
                        'C_k_tau' in f):
                        warnings.warn('Estimating from archived C_k_tau: earlier '
                                      'sweeps time-demeaned the modes and used '
                                      'unmatched zero/positive-lag time windows. '
                                      'Growth rates can be biased; save phi_runs '
                                      'or direct per-run eigenvalue estimates.',
                                      stacklevel=2)
                        warned_covariance=True
                    if eigen_cache is not None and path in eigen_cache:
                        score=eigen_cache[path][key]
                    else:
                        estimates=eigen_indicators(f,params,**(eigen_options or {}))
                        if eigen_cache is not None:
                            eigen_cache[path]=estimates
                        score=estimates[key]
                else:
                    score=dynamic_indicators(
                        f,k_band,min_peak_ratio,
                        max_lag_fraction=max_lag_fraction)[key]
            if np.isfinite(score):
                sums[ib,ix]+=score
                counts[ib,ix]+=1
                matched+=1
        except (OSError,KeyError,ValueError) as exc:
            warnings.warn(f'Cannot read {key} from {path}: {exc}',stacklevel=2)
    result=np.full(shape,np.nan)
    np.divide(sums,counts,out=result,where=counts>0)
    if matched==0:
        required = ('phi_runs or directly saved eigenvalue scalars, or C_k_tau'
                    if key in EIGEN_KEYS else 'C_k_tau and S_k_omega')
        warnings.warn(f'No {key} found for {family}; need {required} in '
                      'per-run NPZ files.',stacklevel=2)
    return result


def pretty_label(key):
    return {
        'C_late': r'Channel order $\mathcal{C}$',
        'entropy_kl_late': r'Compositional $D_{\mathrm{KL}}$',
        'normalized_entropy_segregation_late': r'Normalized $D_{\mathrm{KL}}$',
        'I_occ_late': r'Occupancy variance $I_{\mathrm{occ}}$',
        'oscillation_peak': r'Oscillation peak prominence',
        'oscillation_frequency': r'Oscillation peak $\omega$',
        'oscillation_negative_lobe': r'Negative temporal-correlation lobe',
        'temporal_antisymmetry': r'Cross-lag time asymmetry',
        'Omega_max': r'Empirical $\Omega_{\max}$',
        'r_max': r'Empirical $r_{\max}$',
        'r_at_Omega_max': r'$r$ at $\Omega_{\max}$',
        'k_Omega_max': r'$k$ at $\Omega_{\max}$',
        'k_r_max': r'$k$ at $r_{\max}$',
        'r_complex_max': r'Largest growth of oscillatory mode',
        'Omega_at_r_complex_max': r'$\Omega$ at fastest-growing complex mode',
        'k_r_complex_max': r'$k$ at fastest-growing complex mode',
    }.get(key,key)

def cell_edges(values):
    """Bin edges with given samples at cell centers, for correctly aligned maps."""
    v=np.asarray(values,dtype=float)
    if len(v)<2:
        step=1.0
        return np.array([v[0]-step/2,v[0]+step/2])
    mid=0.5*(v[:-1]+v[1:])
    return np.concatenate(([v[0]-(mid[0]-v[0])],mid,[v[-1]+(v[-1]-mid[-1])]))


def main(argv=None):
    args=parse_args(argv)
    found=[]
    for family in FAMILIES:
        path=args.out/f'{family}_grids.npz'
        if path.exists(): found.append((family,path))
    if not found:
        raise SystemExit('No *_grids.npz found; run sweep or --aggregate-only')
    if args.theory_grid < 5: raise ValueError('--theory-grid must be >=5')
    params=physical_config(args)
    print(f'Theory overlay: sigma={params["sigma"]:g}, rho={params["rho"]}, D={params["D"]}, '
          f'N={params["N"]}, L={params["L"]}, bc={params["bc"]}')

    if args.osc_k_band is not None and (args.osc_k_band[0]<0 or args.osc_k_band[1]<=args.osc_k_band[0]):
        raise ValueError('--osc-k-band must be 0 <= lower < upper')
    if args.osc_min_peak_ratio<=1:
        raise ValueError('--osc-min-peak-ratio must exceed 1')
    if not 0 < args.osc_max_lag_fraction <= 1:
        raise ValueError('--osc-max-lag-fraction must lie in (0,1]')
    # Fail early if a requested k scan has impossible width/spacing.
    if (args.eig_k_range[0] <= 0 or args.eig_k_range[1] <= args.eig_k_range[0] or
        args.eig_step <= 0 or args.eig_bin_width <= 0 or
        args.eig_lag < 1 or args.eig_min_modes < 1 or args.eig_max_condition <= 1 or
        not 0 <= args.eig_min_modulus < 1 or args.eig_min_cycles < 0 or
        any(w<=0 or w>args.eig_k_range[1]-args.eig_k_range[0] for w in args.eig_widths)):
        raise ValueError('Invalid --eig-* options')
    eigen_options=dict(k_range=args.eig_k_range, widths=args.eig_widths,
                       step=args.eig_step, bin_width=args.eig_bin_width,
                       lag_frames=args.eig_lag, min_modes=args.eig_min_modes,
                       max_condition=args.eig_max_condition,
                       min_modulus=args.eig_min_modulus,
                       min_cycles=args.eig_min_cycles)
    fig,axes=plt.subplots(len(found),len(args.keys),
             figsize=(5.*len(args.keys),4.5*len(found)),squeeze=False,
             constrained_layout=True)
    eigen_cache={}  # Avoid repeating a DCT regression for every plotted eigen key.
    for irow,(family,path) in enumerate(found):
        with np.load(path,allow_pickle=False) as data:
            bs=np.asarray(data['beta'],dtype=float)
            xs=np.asarray(data['x'],dtype=float)
            theory_beta=np.linspace(bs[0],bs[-1],args.theory_grid)
            theory_x=np.linspace(xs[0],xs[-1],args.theory_grid)
            theory=None
            longwave_theory=None
            if not args.no_theory:
                theory=theory_data(family,theory_beta,theory_x,params)
                if args.longwave:
                    longwave_theory=theory_data(family,theory_beta,theory_x,params,longwave=True)
                n_stable=np.mean(theory['margin']>0.)
                print(f'{family}: stable fraction on theory grid={n_stable:.3f}; '
                      f'margin range {theory["margin"].min():.4g}..{theory["margin"].max():.4g}')
            for icol,key in enumerate(args.keys):
                ax=axes[irow,icol]
                if key in DYNAMIC_KEYS:
                    values=dynamic_grid(args.out,family,bs,xs,key,
                                        k_band=args.osc_k_band,
                                        min_peak_ratio=args.osc_min_peak_ratio,
                                        max_lag_fraction=args.osc_max_lag_fraction,
                                        eigen_options=eigen_options, params=params,
                                        eigen_cache=eigen_cache)
                elif key in data:
                    values=np.asarray(data[key],dtype=float)
                else:
                    values=None
                if values is None or not np.any(np.isfinite(values)):
                    ax.text(.5,.5,f'{key}\nnot available',ha='center',va='center',transform=ax.transAxes)
                else:
                    expected=(len(bs),len(xs))
                    if values.shape != expected:
                        raise ValueError(f'{path}: {key} shape {values.shape}, expected {expected}')
                    kwargs={}
                    if key=='C_late': kwargs.update(cmap='RdYlGn',vmin=-1,vmax=1)
                    elif key in ('oscillation_peak','oscillation_negative_lobe'):
                        kwargs.update(cmap='magma',vmin=0)
                    elif key == 'entropy_kl_late': kwargs.update(cmap = 'plasma')
                    elif key=='temporal_antisymmetry': kwargs.update(cmap='magma',vmin=0)
                    elif key=='oscillation_frequency': kwargs.update(cmap='plasma',vmin=0)
                    elif key in ('r_max','r_at_Omega_max','r_complex_max'):
                        kwargs.update(cmap='coolwarm') #,norm=CenteredNorm(vcenter=0))
                    elif key in EIGEN_KEYS:
                        kwargs.update(cmap='viridis',vmin=0)
                    else: kwargs.update(cmap='viridis',vmin=0)
                    # Sweep values indexed [beta,x], pcolormesh rows=x cols=beta.
                    im=ax.pcolormesh(cell_edges(bs),cell_edges(xs),
                                     np.ma.masked_invalid(values.T),
                                     shading='flat',**kwargs)
                    fig.colorbar(im,ax=ax,shrink=.85)
                if not args.no_theory:
                    overlay_threshold(ax,family,theory_beta,theory_x,params,
                                      fields=theory,longwave_fields=longwave_theory)
                ax.set_title(f'{family}: {pretty_label(key)}')
                ax.set_xlabel(r'$\beta$')
                ax.set_ylabel(r'$k_o$' if family=='reciprocal' else r'$x$')
                ax.set_xlim(bs[0],bs[-1]); ax.set_ylim(xs[0],xs[-1])
    if not args.no_theory:
        handles=[]
        if any(family=='reciprocal' for family,_ in found):
            handles += [Line2D([0],[0],color='black',lw=2.2,ls='-',
                               label=r'Reciprocal $\lambda_-^{\min}=0$'),
                        Line2D([0],[0],color='black',lw=2.2,ls='--',
                               label=r'Reciprocal $\lambda_+^{\min}=0$')]
        if any(family=='nonreciprocal' for family,_ in found):
            handles += [Line2D([0],[0],color='black',lw=2,ls='-',
                               label='Nonreciprocal onset of stationary instability'),
                        Line2D([0],[0],color='black',lw=2,ls='--',
                               label='Nonreciprocal onset of oscillatory instability'),
                        Line2D([0],[0],color='#7a2778',lw=2,ls=':',
                               label=r'Complex modes: $\min_q\Delta(q)=0$')]
        if args.longwave:
            handles.append(Line2D([0],[0],color='#555555',lw=1.4,ls='-.',
                                  label='Infinite-system $J$ instability threshold'))
            if any(family=='nonreciprocal' for family,_ in found):
                handles.append(Line2D([0],[0],color='#1a7c93',lw=1.5,ls=':',
                                      label=r'Long-wave complex onset: $\Delta(J)=0$'))
        fig.legend(handles=handles,loc='outside lower center',ncol=min(len(handles),3),frameon=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print("Saved", args.save)

    return fig, axes


if __name__ == "__main__":
    main()