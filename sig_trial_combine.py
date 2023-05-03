import numpy as np
import os
import csky as cy
from _paths import PATHS
from _loader import easy_source_list_loader as src_load
from IPython import embed

import matplotlib.pyplot as plt
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials_new", "bg_new")
sig_dir = os.path.join(PATHS.data, "sig_trials_new_2", "sig_new")

#/data/user/jkollek/csky_ehe_stacking/rawout_tests/sig_trials_new/sig_new/for_gamma_3/gamma/2.25/sig/120.0/trials__N_005000_seed_0100_job_run_690.npy

#test_path = "/for_gamma_3/gamma/2.25/sig/120.0/trials__N_005000_seed_0100_job_run_690.npy"

#test_file = sig_dir + test_path

#test = np.load(test_file)

#embed()

print("setup ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_new")
"""
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC79,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86v3_2015,
                                             dir=ana_dir)
"""
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)


cy.CONF['ana'] = ana11

print("load bg")

bg = cy.bk.get_all(
        # disk location
        '{}/bg'.format(bg_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)

print("load sig")



sig = cy.bk.get_all(
        # disk location
        '{}/for_gamma_3/gamma'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
	)


# we need that for some reason, guess it's just a setup
# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]


gamma = list(np.round(np.linspace(1.5,3,7),decimals=4))

#for testing
#cy.hyp.PowerLawFlux(gamma=4)


trs = {g:cy.get_trial_runner(src=cy.sources(ra=src_ra, dec=src_dec),flux=cy.hyp.PowerLawFlux(gamma=g)) for g in gamma}

# now to compute sens and disc

"""
for g in gamma:
	for j in range(4):
		for i in range(len(src_ra)):
			trs[g].sig_injs[j].flux[i] = cy.hyp.PowerLawFlux(gamma=g)
"""


@np.vectorize
def find_n_sig(gamma,beta=0.9, nsigma=None):
    # get signal trials, background distribution, and trial runner
    sig_trials = cy.bk.get_best(sig,gamma,'sig')
    #sig_trials = sig
    #print(sig_trials)
    b = bg
    tr = cy.bk.get_best(trs,gamma)
    # determine ts threshold
    if nsigma is not None:
        ts = b.isf_nsigma(nsigma)
    else:
        ts = b.median()
    # include background trials in calculation
    print("ts: ",ts)
    trials = {0: b.trials}
    trials.update(sig_trials)
    #print(trials)
    #trials = {"nsig":trials}
    #print(trials["0.0"])
    # get number of signal events
    # (arguments prevent additional trials from being run)
    result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=False, trials=trials, n_bootstrap=1)
    # return flux
    return result ,tr.to_dNdE(result, E0=1, unit=1e3)


#sig_trials_gamma_3 = cy.bkget_best(sig,3.0,'sig')
#tr_gamma_3 = cy.bk.get_best(trs,3.0)
#sens = tr_gamma_3.find_n_sig(bg.median(), 0.9, n_sig_step=10, batch_size=500, tol=.05, mp_cpus=5)

sens_res, sens = find_n_sig(gamma)
disc_res, disc = find_n_sig(gamma,beta=0.5, nsigma=5)


# make some tables

table = ""
all_used_signal_params = np.sort(list(sens_res[1]["tss"].keys()))
for _all_used in all_used_signal_params:
    table = table + "{:.1f}".format(_all_used) + " \\\ "

sig_file = open("tables/sig_time_int_table.tex", "w")
n = sig_file.write(table)
sig_file.close()

table = ""
all_used_signal_params = np.sort(list(sens_res[1]["tss"].keys()))
for _all_used in all_used_signal_params[:9]:
    table = table + "{:.1f}".format(_all_used) + " & "
table = table + "{:.1f}".format(all_used_signal_params[10]) + " \\\ "
sig_file = open("tables/sig_time_int_table_2.tex", "w")
n = sig_file.write(table)
sig_file.close()

table = ""
all_used_signal_params = np.sort(list(sens_res[1]["tss"].keys()))
for _all_used in all_used_signal_params[10:-1]:
    table = table + "{:.1f}".format(_all_used) + " & "
table = table + "{:.1f}".format(all_used_signal_params[-1]) + " \\\ "
sig_file = open("tables/sig_time_int_table_3.tex", "w")
n = sig_file.write(table)
sig_file.close()



def num_of_zeros(n):
  s = '{:.16f}'.format(n).split('.')[1]
  return len(s) - len(s.lstrip('0'))

table = ""

for i,_gamma in enumerate(gamma):
    zeros_sens = num_of_zeros(sens[i])
    zeros_disc = num_of_zeros(disc[i])
    _sens = "\\" +"num{"  + "{:.2f}".format(sens[i] * 10**(zeros_sens+1))+"e-{}".format(zeros_sens+1)+"}"
    _disc = "\\" +"num{"  + "{:.2f}".format(disc[i] * 10**(zeros_disc+1))+"e-{}".format(zeros_disc+1)+"}"
    table = table + "{:.2f}".format(_gamma) + " & " + "{:.2f}".format(sens_res[i]["n_sig"]) + " & " + _sens + " & " + "{:.2f}".format(disc_res[i]["n_sig"]) + " & " + _disc + " \\\ "
res_file = open("tables/res_time_int_table.tex", "w")
n = res_file.write(table)
res_file.close()

table = ""
for i,_gamma in enumerate(gamma):
    table = table + "{:.2f}".format(_gamma)
    for item in all_used_signal_params[:10]:
        table = table + " & " + "{}".format(len(sens_res[i]["tss"][item]))
    table = table + " \\\ "
trials_file = open("tables/trials_sig_time_int_table.tex", "w")
n = trials_file.write(table)
trials_file.close()

table = ""
for i,_gamma in enumerate(gamma):
    table = table + "{:.2f}".format(_gamma)
    for item in all_used_signal_params[10:]:
        table = table + " & " + "{}".format(len(sens_res[i]["tss"][item]))
    table = table + " \\\ "
trials_file = open("tables/trials_sig_time_int_table_2.tex", "w")
n = trials_file.write(table)
trials_file.close()


print("sens: ", sens)
print("disc: ", disc)

print("plotting sens, disc")

_fontsize = 10
_markersize = 10
_labelsize = 10
plt.plot(gamma,sens,'x',label=r'sens, $90\%$ at $1\mathrm{TeV}$',markersize=_markersize)
plt.plot(gamma,disc,'x',label=r'disc, $50\%$, $5\sigma$ at $1\mathrm{TeV}$',markersize=_markersize)
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--')
plt.yscale("log")
plt.xlim(1.3,3.2)
plt.xlabel(r'Source spectral index $\gamma$',fontsize=_fontsize)
plt.ylabel(r'$\phi_{1\mathrm{TeV}}[\mathrm{TeV}^{-1}\mathrm{cm}^{-2}\mathrm{s}^{-1}]$',fontsize=_fontsize)
plt.legend(loc="best")
plt.savefig("test_plots/time_int_sens_gfu_gold_9_years_new.pdf")
plt.clf()

print("plotting cdfs")

#sens_res[4]["info"]["n_sigs"]

n_cols = 2
n_rows = len(gamma)


fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

for i,_gamma in enumerate(gamma):
    ax = axs[i,0]
    n_sigs = sens_res[i]["info"]["n_sigs"]
    cls = np.array(sens_res[i]["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = sens_res[i]["info"]["spline"]
    ax.plot(x,spline(x),label="CDF sensitivity spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF sensitivity fit points")
    ax.text(n_sigs[bounds-1]/2.,0.65,r"$\gamma\:=\:{}$".format(_gamma))
    ax.axhline(0.9,linestyle='--',alpha=0.7,label=r'$90\%$')
    ax.set_ylim(0.4,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$')
    ax.set_ylabel(r'CDF')
    ax.legend()

    ax = axs[i,1]
    n_sigs = disc_res[i]["info"]["n_sigs"]
    cls = np.array(disc_res[i]["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = disc_res[i]["info"]["spline"]
    ax.plot(x,spline(x),label="CDF discovery spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF discovery fit points")
    ax.text(n_sigs[bounds-1]/2.,0.3,r"$\gamma\:=\:{}$".format(_gamma))
    ax.axhline(0.5,linestyle='--',alpha=0.7,label=r'$50\%$')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$')
    ax.set_ylabel(r'CDF')
    ax.legend()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_cdf.pdf")
plt.clf()

n_cols = 3
n_rows = 3


fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = np.ravel(axs)

for i,_gamma in enumerate(gamma):
    ax = axs[i]
    n_sigs = sens_res[i]["info"]["n_sigs"]
    cls = np.array(sens_res[i]["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = sens_res[i]["info"]["spline"]
    ax.plot(x,spline(x),label="CDF sensitivity spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF sensitivity fit points")
    ax.text(n_sigs[bounds-1]/2.,0.65,r"$\gamma\:=\:{}$".format(_gamma),fontsize=17)
    ax.axhline(0.9,linestyle='--',alpha=0.7,c="blue",label=r'$90\%$')
    ax.set_ylim(0.4,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    if(i==4 or i==5 or i==6):
        ax.set_xlabel(r'$n_{S}$',fontsize=17)
    if(i==0 or i==3 or i==6):
        ax.set_ylabel(r'CDF',fontsize=17)
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
axs[-1].set_visible(False)
axs[-2].plot([100,101],[0.5,0.5],"-",label=r"CDF sensitivity spline")
axs[-2].plot(100,0.5,"x",label=r"CDF sensitivity fit points") #for the label
axs[-2].plot([100,101],[0.5,0.5],c="blue",linestyle='--',alpha=0.7,label=r'$90\%$')
axs[-2].set_xlim(0,200)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 17},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_cdf_sens.pdf")
plt.clf()


n_cols = 3
n_rows = 3


fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = np.ravel(axs)

for i,_gamma in enumerate(gamma):
    ax = axs[i]
    n_sigs = disc_res[i]["info"]["n_sigs"]
    cls = np.array(disc_res[i]["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = disc_res[i]["info"]["spline"]
    ax.plot(x,spline(x),label="CDF discovery spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF discovery fit points")
    ax.text(n_sigs[bounds-1]/10.,0.8,r"$\gamma\:=\:{}$".format(_gamma),fontsize=17)
    ax.axhline(0.5,linestyle='--',c="blue",alpha=0.7,label=r'$50\%$')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    if(i==4 or i==5 or i==6):
        ax.set_xlabel(r'$n_{S}$',fontsize=17)
    if(i==0 or i==3 or i==6):
        ax.set_ylabel(r'CDF',fontsize=17)
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
axs[-1].set_visible(False)
axs[-2].plot([100,101],[0.5,0.5],"-",label=r"CDF discovery spline")
axs[-2].plot(100,0.5,"x",label=r"CDF discovery fit points") #for the label
axs[-2].plot([100,101],[0.5,0.5],c="blue",linestyle='--',alpha=0.7,label=r'$50\%$')
axs[-2].set_xlim(0,200)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 17},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_cdf_disc.pdf")
plt.clf()


print("Done")
