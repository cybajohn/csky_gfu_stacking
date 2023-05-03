"""
this is time-dependent for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt

from _loader import easy_source_list_loader as src_load
import csky as cy
import histlite as hl

# load sources

# load sources

srcs = src_load()

n_srcs = 20

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)


src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

src_time = [_src['mjd'] for _src in srcs]

test_src = srcs[src_id[9]]

# convert sources to csky_style

src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'])


ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            #'version-004-p00', cy.selections.PSDataSpecs.IC86_2011,
                                            #'version-004-p00', cy.selections.PSDataSpecs.IC86v3_2012_2017,
                                            )

print(ana11.mjd_min,test_src['mjd'],ana11.mjd_max)

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
     }

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=50)

bg = tr_box.get_many_fits(1000, seed=1, mp_cpus=10)

bg = cy.dists.Chi2TSD(bg)

print(bg)

"""
Configuration = {'ana': ana11,
                 'space': "ps",
                 'time': "transient",
                 'sig': 'transient'}

cy.CONF.update(Configuration)

mjd = np.array(test_src['mjd']) - np.array([10]) # Start of box time window
t100 = np.array([10]) # Width of box time window, in days
src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'], deg=False, mjd=mjd, t_100=t100, sigma_t=1*[0]) 

trtr = cy.get_trial_runner(src=src)

bg = trtr.get_many_fits(1000, seed=1, mp_cpus=10)

bg = cy.dists.Chi2TSD(bg)
"""
from IPython import embed
embed()

fig, ax = plt.subplots()

h = bg.get_hist(bins=15)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/time_dep_test_t0_dtmax.pdf")
plt.clf()

#B_space
fig, ax = plt.subplots(1, 1, figsize=(5,4))
for (i, a) in enumerate(ana11):
    print(i)
    hl.plot1d (ax, a.bg_space_param.h, crosses=True, color='k', label='histogram')
    sd = np.linspace (-1, 1, 300)
    ax.plot (sd, a.bg_space_param(sindec=sd), label='spline')
    ax.set_ylim(0)
    ax.set_title(a.plot_key)
    ax.set_xlabel(r'$\sin(\delta)$')
    ax.set_ylabel(r'probability density')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig("test_plots/time_dep_test_space.pdf")
plt.clf()

#S/B_Energy
fig, axs = plt.subplots(1, 4, figsize=(15,3))
gammas = [1,2,3,4]
axs = np.ravel(axs)
for (i, gamma) in enumerate(gammas):
    a = ana11.anas[0]
    ax = axs[i]
    eprm = a.energy_pdf_ratio_model
    ss = dict(zip(eprm.gammas, eprm.ss_hl))
    things = hl.plot2d(ax, ss[gamma].eval(bins=100),
                       vmin=1e-2, vmax=1e2, log=True, cbar=True, cmap='RdBu_r')
    ax.set_title('$\gamma$ = {}'.format(gamma))
    things['colorbar'].set_label(r'$S/B$')
    ax.set_xlabel(r'$\sin(\delta)$')
    ax.set_ylabel(r'$\log_{10}(E/{GeV})$')
plt.tight_layout()
plt.savefig("test_plots/time_dep_test_energy.pdf")
plt.clf()

"""
# t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)

plt.hist(bg.trials["t0"]+bg.trials["dt"]/2,histtype="step",bins=50)
plt.axvline(test_src['mjd'],color="red",label="source pos")
plt.xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
plt.legend(loc="best")
plt.savefig("test_plots/time_dep_t0_test3.pdf")
plt.clf()

#hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

plt.hist(bg.trials["dt"],histtype="step",bins=50)
plt.yscale('log')
plt.xlabel("dt / d")
plt.savefig("test_plots/time_dep_dt_test3.pdf")
plt.clf()


plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=50)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_dt_test3.pdf")
plt.clf()

plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=25)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_dt_test4.pdf")
plt.clf()

plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["ns"],bins=25)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$n_\mathrm{S}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_ns_test3.pdf")
plt.clf()

"""
