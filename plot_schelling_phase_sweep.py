#!/usr/bin/env python3
"""Plot the reciprocal/nonreciprocal phase-diagram maps saved by schelling_phase_sweep.py.

Usage:
 python plot_schelling_phase_sweep.py --out phase_sweep
 python plot_schelling_phase_sweep.py --out phase_sweep --keys C_late I_seg_excess_late I_occ_late
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,default=Path('schelling_phase_sweep'))
p.add_argument('--keys',nargs='+',default=['C_late','I_seg_excess_late','I_occ_late'])
p.add_argument('--save',type=Path,default=None)
a=p.parse_args()

families=['reciprocal','nonreciprocal']
found=[]
for fam in families:
    path=a.out/f'{fam}_grids.npz'
    if path.exists(): found.append((fam,path))
if not found: p.error('No *_grids.npz files found; finish a sweep or run --aggregate-only')

fig,axes=plt.subplots(len(found),len(a.keys),figsize=(4.7*len(a.keys),4*len(found)),
                      squeeze=False,constrained_layout=True)
for ir,(family,path) in enumerate(found):
    with np.load(path,allow_pickle=False) as z:
        xs=z['x']; bs=z['beta']
        for ic,key in enumerate(a.keys):
            ax=axes[ir,ic]
            if key not in z:
                ax.text(.5,.5,f'{key}\nmissing',ha='center',va='center',transform=ax.transAxes)
                continue
            values=z[key]
            image=np.ma.masked_invalid(values)
            kwargs=dict(origin='lower',aspect='auto',interpolation='nearest',
                        extent=[bs[0],bs[-1],xs[0],xs[-1]])
            if key=='C_late': kwargs.update(cmap='RdBu_r',vmin=-1,vmax=1)
            else: kwargs.update(cmap='viridis')
            im=ax.imshow(image.T,**kwargs)
            ax.set_title(f'{family}: {key}')
            ax.set_ylabel('$k_o$' if family=='reciprocal' else '$x$')
            ax.set_xlabel(r'$\beta$')
            fig.colorbar(im,ax=ax,shrink=.85)
if a.save:
    a.save.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.save,dpi=180)
    print('Saved',a.save)
else:plt.show()
