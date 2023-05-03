import numpy as np
import matplotlib.pyplot as plt
from _paths import PATHS
import os

from _loader import easy_source_list_loader as src_load
import csky as cy
import histlite as hl
from IPython import embed
import matplotlib as mpl
import matplotlib.cm as cm

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]

# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)

# load mc, data

ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_new")

ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)


gamma = 2

# get trial runner
tr = cy.get_trial_runner(src=src, ana=ana11,flux=cy.hyp.PowerLawFlux(gamma=gamma))

print('From the TrialRunner:')
llh_model = cy.inspect.get_llh_model(tr, -1)
pdf_ratio_model = cy.inspect.get_pdf_ratio_model(tr, -1)
space_model = cy.inspect.get_space_model(tr, -1)
energy_model = cy.inspect.get_energy_model(tr, -1)

print("llh_model, pdf_ratio_model, space_model, energy_model")

#energy_model.hs_ratio

# bg space pdf time integrated

title = "IC86 2011-2019"

fig, axs = plt.subplots(1,1)
ax = axs
hl.plot1d (ax, ana11[0].bg_space_param.h, crosses=True, color='k', label='histogram')
sd = np.linspace (-1, 1, 300)
ax.plot (sd, ana11[0].bg_space_param(sindec=sd), label='spline')
ax.set_ylim(0)
ax.set_title(title)
ax.set_xlabel(r'$\sin(\delta)$')
ax.set_ylabel(r'probability density')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig("test_plots/bg_space_pdf.pdf")
plt.clf()

# energy pdf ratios 

gamma_min = 1.5
gamma_max = 3.
gamma_steps = 7

gammas = list(np.round(np.linspace(gamma_min,gamma_max,gamma_steps),decimals=4))


fig, axs = plt.subplots(3, 3, figsize=(10,8))
axs = np.ravel(axs)
for (i, gamma) in enumerate(gammas):
    ax = axs[i]
    eprm = ana11[0].energy_pdf_ratio_model
    ss = dict(zip(eprm.gammas, eprm.ss_hl))
    things = hl.plot2d(ax, ss[gamma].eval(bins=100),
                       levels=np.logspace(-2, 2, 16+1),
                       vmin=1e-2, vmax=1e2, log=True, cbar=False, cmap='seismic') # cbar = True
    ax.set_title("$\gamma = {}$".format(str(gamma)))
    #things['colorbar'].set_label(r'$S/B$')
    if(i==4 or i==5 or i==6):
        ax.set_xlabel(r'$\sin(\delta)$',fontsize=14)
    else:
        ax.set_xticklabels([])
    if(i==0 or i==3 or i==6):
        ax.set_ylabel(r'$\log_{10}(E/\mathrm{GeV})$',fontsize=14)
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
plt.tight_layout()
axs[-1].set_visible(False)
cmap_=cm.get_cmap("seismic",16)
im = axs[-1].pcolormesh([[]],vmin=1e-2,vmax=1e2,norm=mpl.colors.LogNorm(),cmap=cmap_)
axs[-2].set_visible(False)
cb_labels = [1e-2,1e-1,1e0,1e1,1e2]
cb=plt.colorbar(im,ax=axs.tolist())
cb.set_label(label=r"$S/B$",size=15)
cb.ax.tick_params(labelsize=14)
plt.savefig("test_plots/energy_pdf_ratio.pdf")
plt.clf()

"""
# sig space pdf

sig_dir = os.path.join(PATHS.data, "sig_trials_new_2", "sig_new")

sig = cy.bk.get_all(
        # disk location
        '{}/for_gamma_3/gamma/2.0/sig/99.0'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        #post_convert=cy.utils.Arrays
    )

print("sig")
embed()


L = tr.get_one_llh(5000)
#L = tr.get_one_llh_from_trial(trial)
L.fit(**tr.fitter_args)
space_eval = cy.inspect.get_space_eval(L, -1, 1) # 1 for sig
SB_space = space_eval(gamma=2)[0]
fig, axs = plt.subplots(1, 1, figsize=(3,3))
hl.plot1d(axs, hl.hist(SB_space, bins=50, log=True), crosses=True)
plt.savefig("test_plots/sig_space_pdf.pdf")
"""


# llh scan time int

mpl.rcParams.update({'font.size': 14})

L = tr.get_one_llh(n_sig=100, poisson=False, seed=100)

scan_ts,mesh = L.scan_ts(ns=np.linspace(0,400,50), gamma=np.linspace(1, 4, 31))
_max = np.argwhere(scan_ts == np.amax(scan_ts))
_max_1 = (mesh[1][_max[0][0]][_max[0][1]] + mesh[1][_max[0][0]+1][_max[0][1]+1]) /2
_max_2 = (mesh[0][_max[0][0]][_max[0][1]] + mesh[0][_max[0][0]+1][_max[0][1]+1]) /2
im = plt.pcolormesh(mesh[0].reshape(50,31),mesh[1].reshape(50,31), scan_ts.reshape(50,31))
plt.plot(_max_2,_max_1,"-",markersize = .1,c="red",label=r"$1\sigma$ contour") #for the label
plt.plot(_max_2,_max_1,"-",markersize = .1,c="blue",label=r"$2\sigma$ contour")
plt.plot(_max_2,_max_1,"x",label=r"max value position")
plt.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
    linewidths=3, colors="red", levels = [np.amax(scan_ts)-1/2])
plt.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
    linewidths=3, colors="blue", levels = [np.amax(scan_ts)-2])
plt.xlabel(r"Signal parameter $n_S$")
plt.ylabel(r"Spectral index $\gamma$")
plt.colorbar(im,label=r"$TS$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/llh_scan_time_int.pdf")
plt.clf()

# llh scan time dep

# load sources

t_max = ana11.mjd_max
t_min = ana11.mjd_min

sources = src_load()

# Check if sources are inside the analysis time frame
srcs_all = [src for src in sources if src["mjd"] <= t_max and src["mjd"] >= t_min]

if len(srcs_all) < len(sources):
    print("Number of possible sources reduced ({} -> {}) due to analysis time frame".format(len(srcs),len(srcs_all)))
    srcs = srcs_all

n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_all = [src["signal"] for src in sources]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)
signals_mask_2 = np.in1d(signals_all,signals_used)

src_id_all = np.reshape(np.argwhere(signals_mask_2 == True), n_srcs)

src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)


nrows, ncols = 3, 4

mpl.rcParams.update({'font.size': 15})

fig, axs = plt.subplots(nrows, ncols, figsize=(13 ,8))
axs = np.ravel(axs)

xticks = [0,10,20,30]
yticks = [1,2,3,4]


for i,_src_id in enumerate(src_id):
    ax = axs[i]
    test_src = srcs[_src_id] #9
    
    # convert sources to csky_style
    
    src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'])
    
    conf_box = {
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': test_src['mjd']},
        'seeder': cy.seeding.UTFSeeder(),
        'sig' : 'tw',
        'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=200, flux=cy.hyp.PowerLawFlux(2.0)),
        }

    tr_uncut = cy.get_trial_runner(conf=conf_box,src=src, ana=ana11, dt_max=200, _fmin_method='minuit')
    
    L_uncut = tr_uncut.get_one_llh(n_sig=10, poisson=False, seed=1)
    
    scan_ts,mesh = L_uncut.scan_ts(ns=np.linspace(0,30,50), gamma=np.linspace(1, 4, 31), t0=test_src["mjd"], dt=200)
    scan_ts = scan_ts.reshape(50,31) 
    im = ax.pcolormesh(mesh[0].reshape(50,31),mesh[1].reshape(50,31), scan_ts)
    _max = np.argwhere(scan_ts == np.amax(scan_ts))
    _max_1 = (mesh[1][_max[0][0]][_max[0][1]] + mesh[1][_max[0][0]+1][_max[0][1]+1]) /2
    _max_2 = (mesh[0][_max[0][0]][_max[0][1]] + mesh[0][_max[0][0]+1][_max[0][1]+1]) /2
    ax.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
        linewidths=1, colors="red", levels = [np.amax(scan_ts)-1/2])
    ax.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
        linewidths=1, colors="blue", levels = [np.amax(scan_ts)-2])
    ax.plot(_max_2,_max_1,"x")
    ax.set_title("Nr. " + str(src_id_all[i]+1))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    #ax.set_box_aspect(1)
    if i == 9 or i == 8 or i == 7 or i == 6:
        ax.set_xlabel(r"$n_S$")
    else:
        ax.set_xticklabels([])
    if i == 0 or i == 4 or i == 8:
        ax.set_ylabel(r"$\gamma$")
    else:
        ax.set_yticklabels([])
axs[-2].plot(_max_2,_max_1,"-",markersize = .1,c="red",label=r"$1\sigma$ contour") #for the label
axs[-2].plot(_max_2,_max_1,"-",markersize = .1,c="blue",label=r"$2\sigma$ contour")
axs[-2].plot(_max_2,_max_1,"x",label=r"max value position")
axs[-2].plot(_max_2,_max_1,"x",c="white")
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 13})
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
axs[-1].set_visible(False)
plt.colorbar(im,ax=axs.tolist(),label=r"$TS$")
plt.savefig("test_plots/llh_scan.pdf")

plt.clf()


print("make 1 plot for show")

test_src = srcs[src_id[0]]

# convert sources to csky_style

src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'])

conf_box = {
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': test_src['mjd']},
        'seeder': cy.seeding.UTFSeeder(),
        'sig' : 'tw',
        'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=200, flux=cy.hyp.PowerLawFlux(2.0)),
}

tr_uncut = cy.get_trial_runner(conf=conf_box,src=src, ana=ana11, dt_max=200, _fmin_method='minuit')

L_uncut = tr_uncut.get_one_llh(n_sig=10, poisson=False, seed=1)

fig = plt.figure(figsize=(9.6,8))

scan_ts,mesh = L_uncut.scan_ts(ns=np.linspace(0,30,50), gamma=np.linspace(1, 4, 31), t0=test_src["mjd"], dt=200)
scan_ts = scan_ts.reshape(50,31)
_max = np.argwhere(scan_ts == np.amax(scan_ts))
_max_1 = (mesh[1][_max[0][0]][_max[0][1]] + mesh[1][_max[0][0]+1][_max[0][1]+1]) /2
_max_2 = (mesh[0][_max[0][0]][_max[0][1]] + mesh[0][_max[0][0]+1][_max[0][1]+1]) /2
im = plt.pcolormesh(mesh[0].reshape(50,31),mesh[1].reshape(50,31), scan_ts.reshape(50,31))
plt.plot(_max_2,_max_1,"-",markersize = .1,c="red",label=r"$1\sigma$ contour") #for the label
plt.plot(_max_2,_max_1,"-",markersize = .1,c="blue",label=r"$2\sigma$ contour")
plt.plot(_max_2,_max_1,"x",label=r"max value position")
plt.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
    linewidths=3, colors="red", levels = [np.amax(scan_ts)-1/2])
plt.contour(scan_ts.transpose(),extent=[mesh[0][0][0],mesh[0][-1][-1],mesh[1][0][0],mesh[1][-1][-1]],
    linewidths=3, colors="blue", levels = [np.amax(scan_ts)-2])
plt.title("Source Nr. " + str(src_id_all[0]+1))
plt.xlabel(r"Signal parameter $n_S$")
plt.ylabel(r"Spectral index $\gamma$")
plt.colorbar(im,label=r"$TS$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/llh_scan_1.pdf")
plt.clf()

print("Done")

"""
fig, axs = plt.subplots(1,1)
ax = axs
sd = np.linspace (-1, 1, 300)

for i in range(1,200):
    conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': srcs[src_id]['mjd'],"gamma": 2.,'dt': i},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=srcs[src_id]['mjd'], dt=i, flux=cy.hyp.PowerLawFlux(2.)),
    }
    tr = cy.get_trial_runner(conf=conf_box, ana=ana11, src=src)
    L = tr.get_one_llh()
    embed()
    hist = L.llh_model.pdf_ratio_model.models[0].space_model.bg_param.h
    hl.plot1d (ax, hist, crosses=True)
ax.set_ylim(0)
ax.set_title(title)
ax.set_xlabel(r'$\sin(\delta)$')
ax.set_ylabel(r'probability density')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig("test_plots/bg_space_pdf_time_dep.pdf")

"""
