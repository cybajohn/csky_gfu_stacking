import os
import json
from glob import glob
import gzip
import numpy as np
import healpy as hp
import histlite as hl

from IPython import embed

import matplotlib.pyplot as plt

import csky as cy

from _paths import PATHS
import _loader

import math

print("Loading sources")
sources = _loader.easy_source_list_loader()

# get source positions
ras = [src["ra"] for src in sources]
decs = [src["dec"] for src in sources]
signal = [src["signal"] for src in sources]
print("signalness:")
print(signal)
#embed()


# use csky for good measures
src = cy.utils.Sources(ra = ras, dec = decs)

# low resolution for bigger points
print("Creating skymap")
NSIDE = 512
src_map = hl.heal.hist(NSIDE, src.dec, src.ra)

print("Plotting skymap")
fig, ax = plt.subplots (subplot_kw=dict (projection='aitoff'))
sp = cy.plotting.SkyPlotter(pc_kw=dict(cmap='Greys', vmin=0))
mesh, cb = sp.plot_map(ax, np.where(src_map.map>0, src_map.map, np.nan), n_ticks=2) #2
# get positions from map using precoded methods so I dont mess up
bf_th, bf_phi = hp.pix2ang(NSIDE, np.where(src_map.map>0))
x,y = sp.thetaphi_to_mpl(bf_th,bf_phi)
cb.remove()
im = ax.scatter(x,y,c=np.array(signal),s=12,cmap='viridis',vmin=0,vmax=1)
kw = dict(color='.5', alpha=.5)
sp.plot_gp(ax, lw=.5, **kw)
sp.plot_gc(ax, **kw)
ax.grid(**kw)
cbar = fig.colorbar(im,orientation="horizontal")
cbar.set_label(r"Probability of being of astrophysical origin")
plt.tight_layout()
plot_name = "test_plots/gfu_gold_skymap.pdf"
plt.savefig(plot_name)

print("Saved skymap under {}".format(plot_name))

plt.clf()
sin_dec = np.sin(np.array([src["dec"] for src in sources]))
bins = np.linspace(-1,1,11)
plt.hist(sin_dec,bins)
plt.savefig("test_plots/gold_sindec_hist.pdf")
plt.clf()

# again for all golds
print("Loading sources")
sources = _loader.easy_source_list_loader(name="all_golds.json")

# get source positions
ras = [src["ra"] for src in sources]
decs = [src["dec"] for src in sources]
signal = [src["signal"] for src in sources]
#embed()


# use csky for good measures
src = cy.utils.Sources(ra = ras, dec = decs)

# low resolution for bigger points
print("Creating skymap")
NSIDE = 512
src_map = hl.heal.hist(NSIDE, src.dec, src.ra)

print("Plotting skymap")
fig, ax = plt.subplots (subplot_kw=dict (projection='aitoff'))
sp = cy.plotting.SkyPlotter(pc_kw=dict(cmap='Greys', vmin=0))
mesh, cb = sp.plot_map(ax, np.where(src_map.map>0, src_map.map, np.nan), n_ticks=2) #2
# get positions from map using precoded methods so I dont mess up
bf_th, bf_phi = hp.pix2ang(NSIDE, np.where(src_map.map>0))
x,y = sp.thetaphi_to_mpl(bf_th,bf_phi)
cb.remove()
im = ax.scatter(x,y,c=np.array(signal),s=12,cmap='viridis',vmin=0,vmax=1)
kw = dict(color='.5', alpha=.5)
sp.plot_gp(ax, lw=.5, **kw)
sp.plot_gc(ax, **kw)
ax.grid(**kw)
cbar = fig.colorbar(im,orientation="horizontal")
cbar.set_label(r"Probability of being of astrophysical origin")
plt.tight_layout()
plot_name = "test_plots/all_gold_skymap.pdf"
plt.savefig(plot_name)

print("Saved skymap under {}".format(plot_name))

plt.clf()
sin_dec = np.sin(np.array([src["dec"] for src in sources]))
bins = np.linspace(-1,1,11)
plt.hist(sin_dec,bins)
plt.savefig("test_plots/all_gold_sindec_hist.pdf")

plt.clf()



# for sources in time-dep only


ana11 = cy.get_analysis(cy.selections.repo,
                        'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                       )

# load sources

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs =  _loader.easy_source_list_loader()

# Check if sources are inside the analysis time frame
srcs_all = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

if len(srcs_all) < len(srcs):
    print("Number of possible sources reduced ({} -> {}) due to analysis time frame".format(len(srcs),len(srcs_all)))
    srcs = srcs_all

n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)

src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

ras = [srcs[item]["ra"] for item in src_id]
decs = [srcs[item]["dec"] for item in src_id]
signal = [srcs[item]["signal"] for item in src_id]


# use csky for good measures
src = cy.utils.Sources(ra = ras, dec = decs)

# low resolution for bigger points
print("Creating time_dep skymap")
NSIDE = 512
src_map = hl.heal.hist(NSIDE, src.dec, src.ra)

print("Plotting skymap")
fig, ax = plt.subplots (subplot_kw=dict (projection='aitoff'))
sp = cy.plotting.SkyPlotter(pc_kw=dict(cmap='Greys', vmin=0))
mesh, cb = sp.plot_map(ax, np.where(src_map.map>0, src_map.map, np.nan), n_ticks=2) #2
# get positions from map using precoded methods so I dont mess up
bf_th, bf_phi = hp.pix2ang(NSIDE, np.where(src_map.map>0))
x,y = sp.thetaphi_to_mpl(bf_th,bf_phi)
cb.remove()
im = ax.scatter(x,y,c=np.array(signal),s=12,cmap='viridis',vmin=0,vmax=1)
kw = dict(color='.5', alpha=.5)
sp.plot_gp(ax, lw=.5, **kw)
sp.plot_gc(ax, **kw)
ax.grid(**kw)
cbar = fig.colorbar(im,orientation="horizontal")
cbar.set_label(r"Probability of being of astrophysical origin")
plt.tight_layout()
plot_name = "test_plots/gfu_gold_skymap_time_dep.pdf"
plt.savefig(plot_name)

print("Saved time_dep skymap under {}".format(plot_name))

plt.clf()

