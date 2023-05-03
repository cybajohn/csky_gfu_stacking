"""
this is time-dependent for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

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

sus_src = srcs[src_id[11]]
comp_src = srcs[src_id[0]]

# convert sources to csky_style

src_sus = cy.utils.Sources(ra=sus_src['ra'], dec=sus_src['dec'])
src_comp = cy.utils.Sources(ra=comp_src['ra'], dec=comp_src['dec'])

ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            #'version-004-p00', cy.selections.PSDataSpecs.IC86_2011,
                                            #'version-004-p00', cy.selections.PSDataSpecs.IC86v3_2012_2017,
                                            )

conf_box = {
    'time': 'utf',
    'box': True,
    'seeder': cy.seeding.UTFSeeder(),
    'dt_max': 20,
     }

tr_box_sus = cy.get_trial_runner(conf_box, src=src_sus, ana=ana11)

bg_sus = cy.dists.Chi2TSD(tr_box_sus.get_many_fits(10000, seed=1,mp_cpus=10))

print(bg_sus)

#from IPython import embed
#embed()

tr_box_comp = cy.get_trial_runner(conf_box, src=src_comp, ana=ana11)
bg_comp = cy.dists.Chi2TSD(tr_box_comp.get_many_fits(10000, seed=1,mp_cpus=10))

print(bg_comp)

fig, ax = plt.subplots()

h = bg_sus.get_hist(bins=30)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg_sus.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg_sus.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg_sus.ndof, bg_sus.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/time_dep_sus_1.pdf")
plt.clf()

# t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)

plt.hist(bg_sus.trials["t0"]+bg_sus.trials["dt"]/2,histtype="step",bins=50)
#plt.axvline(test_src['mjd'],color="red",label="source pos")
plt.xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
#plt.legend(loc="best")
plt.savefig("test_plots/time_dep_sus_2.pdf")
plt.clf()

#hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

plt.hist(bg_sus.trials["dt"],histtype="step",bins=50)
plt.yscale('log')
plt.xlabel("dt / d")
plt.savefig("test_plots/time_dep_sus_3.pdf")
plt.clf()

plt.hist2d(bg_sus.trials["t0"]+bg_sus.trials["dt"]/2,bg_sus.trials["ns"],bins=50, norm=LogNorm())
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$n_\mathrm{S}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_sus_4.pdf")
plt.clf()


fig, ax = plt.subplots()

h = bg_comp.get_hist(bins=30)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg_comp.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg_comp.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg_comp.ndof, bg_comp.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/time_dep_comp_1.pdf")
plt.clf()

# t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)

plt.hist(bg_comp.trials["t0"]+bg_comp.trials["dt"]/2,histtype="step",bins=50)
#plt.axvline(test_src['mjd'],color="red",label="source pos")
plt.xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
#plt.legend(loc="best")
plt.savefig("test_plots/time_dep_comp_2.pdf")
plt.clf()

#hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

plt.hist(bg_comp.trials["dt"],histtype="step",bins=50)
plt.yscale('log')
plt.xlabel("dt / d")
plt.savefig("test_plots/time_dep_comp_3.pdf")
plt.clf()

plt.hist2d(bg_comp.trials["t0"]+bg_comp.trials["dt"]/2,bg_comp.trials["ns"],bins=50, norm=LogNorm())
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$n_\mathrm{S}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_comp_4.pdf")
plt.clf()


"""
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
