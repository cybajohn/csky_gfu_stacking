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

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

srcs = src_load()

bg_dir = os.path.join(PATHS.data, "txs_bg_trials_time_dep", "bg", "src")

src_id = os.listdir(bg_dir)

n_srcs = len(src_id)

n_cols = 1
n_rows = 1
fontsize = 10
n_bins = 50
max_bin = 40
_bins = np.linspace(0, max_bin, n_bins)

#from IPython import embed
#embed()

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

all_bg = []
all_norms = []
all_values = []

for i,src in enumerate(src_id):
    bg = cy.bk.get_all(
        # disk location
        '{}/{}'.format(bg_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)
    all_bg.append(bg)
    all_values.extend(bg.values)
    all_norms.append(bg.get_hist(bins=_bins).integrate().values)

    row_index = int(np.floor(i/n_cols))
    col_index = int(i%n_cols)
    print(row_index)
    print(col_index)
    ax = axs
    #ax.set_box_aspect(1)
    
    sigma_5 = bg.isf_nsigma(5)
    sigma_3 = bg.isf_nsigma(3)
    h = bg.get_hist(bins=_bins)
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))
    
    x = h.centers[0]
    norm = h.integrate().values
    ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
    ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_xlabel(r'TS')
    ax.set_ylabel(r'number of trials')
    ax.legend()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

plt.tight_layout()
plt.savefig("test_plots/txs_9_years_gfu_gold_time_dep_bg.pdf")
plt.clf()


n_cols = 4
n_rows = len(all_bg)
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

all_time_mids = np.concatenate([b.trials["t0"] for b in all_bg])
all_time_mids_min = np.amin(all_time_mids)
all_time_mids_max = np.amax(all_time_mids)
all_time_length = np.concatenate([b.trials["dt"] for b in all_bg])
all_time_length_min = np.amin(all_time_length)
all_time_length_max = np.amax(all_time_length)
all_ns = np.concatenate([b.trials["ns"] for b in all_bg])
all_ns_min = np.amin(all_ns)
all_ns_max = np.amax(all_ns)

n_bins = 50

plot_3_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
               np.linspace(all_time_length_min,all_time_length_max,n_bins+1)]

plot_4_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
               np.linspace(all_ns_min,all_ns_max,n_bins+1)]



for i,bg in enumerate(all_bg):
    ax = axs[0]
    # t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)
    # when box_mode center (def), t0 is the middle of the box
    ax.hist(bg.trials["t0"],histtype="step",bins=50)
    #ax.axvline(srcs[int(src_id[i])]['mjd'],color="red",label="ra: {}, dec: {}".format(srcs[int(src_id[i])]["ra"],srcs[int(src_id[i])]["dec"]))
    ax.text(x=56500,y=5000,s="src pos"+"\n"+ "ra: {}, dec: {}".format(np.round(srcs[int(src_id[i])]["ra"],decimals=3),np.round(srcs[int(src_id[i])]["dec"],decimals=3)))
    ax.set_xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
    #ax.legend()
    #plt.savefig("test_plots/time_dep_t0_test3.pdf")
    #plt.clf()
    
    #hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)
    
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax = axs[1]
    ax.hist(bg.trials["dt"],histtype="step",bins=50)
    ax.set_yscale('log')
    ax.set_xlabel("dt / d")
    #plt.savefig("test_plots/time_dep_dt_test3.pdf")
    #plt.clf()
    
    #ax = axs[i,2]
    #im = ax.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=50)
    #ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    #ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    #plt.colorbar(im,ax)
    #plt.savefig("test_plots/time_dep_d0_dt_test3.pdf")
    #plt.clf()

    ax = axs[2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.hist2d(bg.trials["t0"],bg.trials["dt"],bins=plot_3_bins, norm=LogNorm())
    ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[-1],ax, orientation='vertical', pad=0.1)
    #plt.savefig("test_plots/time_dep_d0_dt_test4.pdf")
    #plt.clf()

    ax = axs[3]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    ax.hist2d(bg.trials["t0"],bg.trials["ns"],bins=plot_4_bins, norm=LogNorm())
    ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    ax.set_ylabel(r"$n_\mathrm{S}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')

plt.tight_layout()
plt.savefig("test_plots/txs_9_years_gfu_gold_time_dep_bg_timewindows.pdf")
plt.clf()


"""

fig, ax = plt.subplots()
values = np.sum([bgr.get_hist(bins=_bins).get_values() for bgr in all_bg],axis=0)/20
embed()
h = hl.hist_direct(all_values,bins=_bins)/20
embed()
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
bg_sum = np.sum(all_norm * [bgr.pdf(x) for bgr in all_bg],axis=0)/20
embed()
ax.semilogy(x, bg_sum, lw=1, ls='--',
        label=r'mean')
ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_mean.pdf")
plt.clf()

"""

"""
print(bg)
from IPython import embed
embed()

sigma_5 = bg.isf_nsigma(5)
sigma_3 = bg.isf_nsigma(3)

fontsize = 15

#fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(13,5))

h = bg.get_hist(bins=50)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8,label=r'$5\sigma$')
ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5,label=r'$3\sigma$')
ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
	item.set_fontsize(fontsize)
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_bg.pdf")
plt.clf()
"""
