import numpy as np
import os
import csky as cy
from _paths import PATHS
from _loader import source_list_loader

import matplotlib.pyplot as plt
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials", "bg")
sig_dir = os.path.join(PATHS.data, "sig_trials", "sig")
"""
print("setup ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC79,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86v3_2015,
                                             dir=ana_dir)
cy.CONF['ana'] = ana11
"""
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
        '{}/for_gammas/gamma'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
        )

#sig_best = cy.bk.get_best('')

from IPython import embed
embed()


# we need that for some reason, guess it's just a setup
# load sources
"""
names = source_list_loader()

src_ra = []
src_dec = []

for name in names:
        srcs = source_list_loader(name)
        srcs_ra = [src["ra"] for src in srcs[name]]
        srcs_dec = [src["dec"] for src in srcs[name]]
        src_ra.extend(srcs_ra)
        src_dec.extend(srcs_dec)

trs = cy.get_trial_runner(src=cy.sources(ra=src_ra, dec=src_dec))


"""

fig, ax = plt.subplots()

h = bg.get_hist(bins=50)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

_n_trials = len(sig[2.0]['sig'][10.7]["ts"])
h_sig = hl.hist(sig[2.0]['sig'][10.7]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials, $\gamma=2$, n_inj= {}'.format(_n_trials,10.7))

h_sig = hl.hist(sig[2.0]['sig'][0.0]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials, $\gamma=2$, n_inj= {}'.format(_n_trials,0.0))

h_sig = hl.hist(sig[2.0]['sig'][16.0]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials, $\gamma=2$, n_inj= {}'.format(_n_trials,16.0))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/test3.pdf")
plt.clf()


fig, ax = plt.subplots()

h = bg.get_hist(bins=50)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

_n_trials = len(sig[2.0]['sig'][10.7]["ts"])
h_sig = hl.hist(sig[1.0]['sig'][10.7]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label=r'{} sig trials, $\gamma=1$, n_inj= {}'.format(_n_trials,10.7))

h_sig = hl.hist(sig[2.0]['sig'][10.7]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label=r'{} sig trials, $\gamma=2$, n_inj= {}'.format(_n_trials,10.7))

h_sig = hl.hist(sig[3.0]['sig'][10.7]["ts"],bins=50)
hl.plot1d(ax, h_sig, crosses=True, label=r'{} sig trials, $\gamma=3$, n_inj= {}'.format(_n_trials,10.7))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/test4.pdf")
plt.clf()



def plot_sig(sig,bg,nsigma,gamma_list,sig_list,name):
	fig, ax = plt.subplots()
	
	h = bg.get_hist(bins=50)
	norm = h.integrate().values
	hl.plot1d(ax, h/norm, crosses=True, label='{} bg trials'.format(bg.n_total))
	
	x = h.centers[0]
	norm = h.integrate().values
	ax.semilogy(x, bg.pdf(x), lw=1, ls='--',
        	    label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

	grays = 0.4
	for sigma in nsigma:
        	grays+=0.1
        	ax.axvline(bg.isf_nsigma(sigma),alpha=0.5,color=str(grays),ls='--',label=r'{}$\sigma$'.format(sigma))
	ax.axvline(bg.median(),alpha=0.5,ls='--',label=r'median(bg)')
	#sort list for aesthetics	
	sig_list = [sig_list[i] for i in np.argsort(sig_list)]
	
	for gamma in gamma_list:
        	for n_sig in sig_list:
                	h_sig = hl.hist(sig[gamma]['sig'][n_sig]["ts"],bins=50)
                	norm = h_sig.integrate()
                	_n_trials = len(sig[gamma]['sig'][n_sig]["ts"])
               		hl.plot1d(ax, h_sig/norm, crosses=True, label=r'{} sig trials, $\gamma={}$, n_inj= {}'.format(_n_trials,gamma,n_sig))
	
	ax.set_xlabel(r'TS')
	ax.set_ylabel(r'pdf')
	ax.legend()
	plt.tight_layout()
	plt.savefig(name)
	plt.clf()
	return


fig, ax = plt.subplots()

h = bg.get_hist(bins=50)
norm = h.integrate().values
hl.plot1d(ax, h/norm, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

nsigma=[3,4,5]
grays = 0.4
for sigma in nsigma:
	grays+=0.1
	ax.axvline(bg.isf_nsigma(sigma),alpha=0.5,color=str(grays),ls='--',label=r'{}$\sigma$'.format(sigma))
ax.axvline(bg.median(),alpha=0.5,ls='--',label=r'median(bg)')

gamma_list=[2.0]
sig_list = list(sig[2.0]['sig'].keys())

sig_list = [sig_list[i] for i in np.argsort(sig_list)]
sig_list = sig_list[::2]

for gamma in gamma_list:
	for n_sig in sig_list:
		h_sig = hl.hist(sig[gamma]['sig'][n_sig]["ts"],bins=50)
		norm = h_sig.integrate()
		_n_trials = len(sig[gamma]['sig'][n_sig]["ts"])
		hl.plot1d(ax, h_sig/norm, crosses=True, label=r'{} sig trials, $\gamma={}$, n_inj= {}'.format(_n_trials,gamma,n_sig))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'pdf')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/test5.pdf")
plt.clf()

nsigma=[3,4,5]
sig_list=[16.0]
gamma_list=[1.5,2.0,2.5,3.0,3.5]
plot_sig(sig,bg,nsigma,gamma_list,sig_list,name="test_plots/test6.pdf")
