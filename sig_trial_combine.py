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

print("setup ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC79,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86v3_2015,
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
        '{}/sig'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
	)

# we need that for some reason, guess it's just a setup
# load sources

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

# now to compute sens and disc

#@np.vectorize
def find_n_sig(beta=0.9, nsigma=None):
    # get signal trials, background distribution, and trial runner
    #sig_trials = cy.bk.get_best(sig,'nsig')
    sig_trials = sig
    #print(sig_trials)
    b = bg
    tr = trs
    # determine ts threshold
    if nsigma is not None:
        ts = b.isf_nsigma(nsigma)
    else:
        ts = b.median()
    # include background trials in calculation
    trials = {0: b.trials}
    trials.update(sig_trials)
    #print(trials)
    #trials = {"nsig":trials}
    #print(trials["0.0"])
    # get number of signal events
    # (arguments prevent additional trials from being run)
    result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=False, trials=trials, n_bootstrap=1)
    # return flux
    return result, tr.to_E2dNdE(result, E0=100, unit=1e3)


a, sens = find_n_sig()
from IPython import embed
embed()

print("sens and disc:", find_n_sig())
print("Done")
