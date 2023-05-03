"""
To plot energy, etc.
"""
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl
import os
from glob import glob
import csky as cy

from _loader import easy_source_list_loader as src_load
from _paths import PATHS

#embed()

version = "version-004-p00"

grl_inpath = "../../../../data/ana/analyses/ps_tracks/{}/GRL".format(version)

# only IC86
grl_npy_files = list(glob(os.path.join(grl_inpath,'IC86_*_exp.npy')))

livetimes = 0

for data_file in grl_npy_files:
    file_name = os.path.basename(data_file)
    print("Working with {}".format(file_name))
    data = np.load(data_file)
    livetimes += np.sum(data['livetime'])

livetimes = livetimes * 24. * 60. * 60.

year = 60**2 * 24 * 365

def mc_weight(mc, live):
    return 1.36*mc['ow']*(live*((mc['trueE']/1e5)**(-2.37))*(10**(-18)))

my_mc_file = os.path.join(PATHS.data, "cleaned_datasets_new", "IC86_2016_MC.npy")
og_mc_file = os.path.join('/data','ana','analyses','ps_tracks','version-004-p00','IC86_2016_MC.npy') 


my_mc = np.load(my_mc_file)
og_mc = np.load(og_mc_file)

#embed()

test = np.in1d(og_mc["trueAzi"],my_mc["trueAzi"])
test2 = np.in1d(og_mc["trueZen"],my_mc["trueZen"])
test3 = np.in1d(og_mc["trueE"],my_mc["trueE"])
test4 = test*test2*test3
gold_events = np.load(og_mc_file)
gold_events = gold_events[~test4]


ana_dir = os.path.join(PATHS.data, "ana_cache", "bg")

ana11 = cy.get_analysis(cy.selections.repo,
                        'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                       dir=ana_dir)

# load sources

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs_all = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

srcs_dec = [src["dec"] for src in srcs_all]
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12,5))

# set font size
mpl.rcParams.update({'font.size': 20}) #15

_bins = np.linspace(-1,1,11)
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.hist(np.sin(srcs_dec), bins = _bins)
ax.set_xlabel(r"$\sin{(\delta)}$",fontsize=18)
ax.set_ylabel(r"Number of events",fontsize=18)
ax.set_xticks(x_ticks)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)


ax = axs[1]
ax.hist(np.sin(gold_events['dec']),weights=mc_weight(gold_events, livetimes), bins=_bins)
ax.set_xlabel(r"$\sin{(\delta)}$",fontsize=18)
ax.set_xticks(x_ticks)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("test_plots/gfu_gold_comp.pdf")
plt.clf()

values = plt.hist(np.sin(gold_events['dec']),weights=mc_weight(gold_events, livetimes))
plt.xlabel(r"$\sin{(\delta)}$")
plt.ylabel(r"Number of events")
plt.savefig("test_plots/mc_gold_sindec_hist.pdf")
plt.clf()

print("finished mc_gold_sindec_hist.pdf")
#embed()

print("make energy comp")

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12,5))

srcs_energy = [src["energy"] for src in srcs_all]
_bins = np.linspace(0,np.amax(gold_events['trueE']),100)

ax = axs[0]
ax.hist(srcs_energy, bins=np.logspace(1,4))
ax.set_xscale('log')
ax.set_xlabel(r"Energy in $\mathrm{TeV}$",fontsize=18)
ax.set_ylabel(r"Number of events",fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)


ax = axs[1]
ax.hist(gold_events['trueE']/1e3,weights=mc_weight(gold_events, livetimes), bins=np.logspace(1,4))
ax.set_xscale('log')
ax.set_xlabel(r"Energy in $\mathrm{TeV}$",fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("test_plots/gfu_gold_energy_comp.pdf")
plt.clf()


fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(20,10))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

_norm = Normalize(-9,1)

ax = axs[0,1]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins)
H = np.log10(H)
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
#ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
ax.set_title(r"MC without GFU-gold")
#plt.colorbar(im,ax=ax)

ax = axs[0,0]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins)
H = np.log10(H)
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
#ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
ax.set_ylabel(r"$\log_{10}(\mathrm{trueE})\:/\:\mathrm{GeV}$")
ax.set_title(r"GFU-gold events")
#plt.colorbar(im,ax=ax)

ax = axs[0,2]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins)
H = np.log10(H)
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
#ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
ax.set_title(r"Original MC")
#axes = [axs[0,0],axs[0,1],axs[0,2]]
plt.colorbar(im,ax=axs[:1].ravel().tolist(),label=r"$\log_{10}{(\#\mathrm{events})}$")

_norm = Normalize(0,5)

ax = axs[1,1]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins)
H[(H == 0)] = np.nan
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#plt.colorbar(im,ax=ax)

ax = axs[1,0]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins)
H[(H == 0)] = np.nan
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
ax.set_ylabel(r"$\log_{10}(\mathrm{trueE})\:/\:\mathrm{GeV}$")
#plt.colorbar(im,ax=ax)

ax = axs[1,2]
ax.set_box_aspect(1)
H , xedges, yedges= np.histogram2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins)
H[(H == 0)] = np.nan
X, Y = np.meshgrid(xedges, yedges)
im = ax.pcolormesh(X, Y, H.T, norm=_norm)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\mathrm{true}\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#axes = [axs[1,0],axs[1,1],axs[1,2]]
plt.colorbar(im,ax=axs[1:].ravel().tolist(),label=r"$\#\mathrm{events}$")


plt.savefig("test_plots/cleaned_mc_energy_test.pdf")
plt.clf()


fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]
"""
ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins)
#im_1 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")

"""

ax = axs[1]
ax.set_box_aspect(1)

im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins)
#im_2 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
plt.colorbar(im[3],ax=ax)


ax = axs[2]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins)
#im_3 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
plt.colorbar(im[3],ax=ax)

test = np.in1d(og_mc["trueAzi"],my_mc["trueAzi"])
test2 = np.in1d(og_mc["trueZen"],my_mc["trueZen"])
test3 = np.in1d(og_mc["trueE"],my_mc["trueE"])
test4 = test*test2*test3
gold_events = np.load(og_mc_file)
gold_events = gold_events[~test4]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins)
#im_1 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")


#cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
#plt.colorbar(im[3], cax=cax, **kw)
plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_wgold.pdf")
plt.clf()


mpl.rcParams.update({'font.size': 15})

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(13,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins)#, norm=LogNorm())
#im_1 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")
plt.colorbar(im[3],ax=ax)

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins)#, norm=LogNorm())
#im_2 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
#plt.colorbar(im[3], cax=cax, **kw)
plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy.pdf")
plt.clf()

"""

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(13,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")
#plt.colorbar(im[3],ax=ax)

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
plt.colorbar(im[3], cax=cax, **kw)
#plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_lognorm.pdf")
plt.clf()

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#plt.colorbar(im[3],ax=ax)

ax = axs[2]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
plt.colorbar(im[3], cax=cax, **kw)
#plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_lognorm_wgold.pdf")
plt.clf()


fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year),bins=bins)
im_1 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year),bins=bins)
im_2 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#plt.colorbar(im[3],ax=ax)



ax = axs[2]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year),bins=bins)
im_3 = np.sum(im[0].reshape(400))
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
plt.colorbar(im[3], cax=cax, **kw)
#plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_wgold.pdf")
plt.clf()

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

im_sum = im_3

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year)/im_sum,bins=bins)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year)/im_sum,bins=bins)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#plt.colorbar(im[3],ax=ax)

ax = axs[2]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year)/im_sum,bins=bins)
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
plt.colorbar(im[3], cax=cax, **kw)
#plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_wgold_normed.pdf")
plt.clf()

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,5))
bins = (np.linspace(-1,1,21),np.linspace(2,9,21))
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(gold_events['trueDec']),np.log10(gold_events['trueE']),weights=mc_weight(gold_events,year)/im_sum,bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
ax.set_ylabel(r"$\log_{10}(E\:/\:\mathrm{GeV})$")

ax = axs[1]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(my_mc['trueDec']),np.log10(my_mc['trueE']),weights=mc_weight(my_mc,year)/im_sum,bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
#plt.colorbar(im[3],ax=ax)

ax = axs[2]
ax.set_box_aspect(1)
im = ax.hist2d(np.sin(og_mc['trueDec']),np.log10(og_mc['trueE']),weights=mc_weight(og_mc,year)/im_sum,bins=bins, norm=LogNorm())
ax.set_xticks(x_ticks)
ax.set_xlabel(r"$\sin(\delta)$")
#ax.set_ylabel(r"$\log_{10}(E)\:/\:\mathrm{GeV}$")
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
plt.colorbar(im[3], cax=cax, **kw)
#plt.colorbar(im[3],ax=ax)


plt.savefig("test_plots/cleaned_mc_energy_lognorm_wgold_normed.pdf")
plt.clf()

"""
