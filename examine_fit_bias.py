import numpy as np
import os
import csky as cy
from _paths import PATHS
from IPython import embed
from _loader import easy_source_list_loader as src_load

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import histlite as hl


bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran_new", "bg", "src")

src_id = os.listdir(bg_dir)

sig_trials_dir = os.path.join(PATHS.data, "sig_trials_time_dep_t0_dt_gamma_ran_new")

sig_dir = cy.utils.ensure_dir('{}/sig/time_dep/gamma/2.0/src'.format(sig_trials_dir))

n_cols = 2
n_rows = len(src_id)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

for i,src in enumerate(src_id):
    sig = cy.bk.get_all(
        # disk location
        '{}/{}'.format(sig_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
    )
    sig_ns = list(sig["sig"].keys())
    sig_gamma_mean = [np.mean(sig["sig"][ns].gamma) for ns in sig_ns]
    sig_gamma_std  = [np.std(sig["sig"][ns].gamma) for ns in sig_ns]
    sig_ns_mean = [np.mean(sig["sig"][ns].ns) for ns in sig_ns]
    sig_ns_std = [np.std(sig["sig"][ns].ns) for ns in sig_ns]
    gamma = 2
    
    ax = axs[i,0]
    ax.axhline(gamma,ls='--',color='cyan')
    ax.errorbar(sig_ns, sig_gamma_mean, yerr=sig_gamma_std, fmt='.')
    ax.set_xlabel(r'$n_S$')
    ax.set_ylabel(r'$\hat\gamma$',rotation='horizontal')
    
    ax = axs[i,1]
    ns_norm = np.linspace(0,np.amax(sig_ns),100)
    ax.plot(ns_norm,ns_norm,linestyle='--')
    ax.errorbar(sig_ns, sig_ns_mean, yerr=sig_ns_std, fmt='.')
    ax.set_xlabel(r'$n_S$')
    ax.set_ylabel(r'$\hat{n_S}$',rotation='horizontal')
 
fig.tight_layout()
plt.savefig('test_plots/sig_trials_time_dep_fit_bias.pdf')
plt.clf()

