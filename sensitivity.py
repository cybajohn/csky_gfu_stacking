import numpy as np
import os
from _paths import PATHS

import matplotlib.pyplot as plt

from _loader import easy_source_list_loader as src_load

import csky as cy
import histlite as hl

# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]


# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)

# load bg

ana11 = cy.get_analysis(cy.selections.repo,
                        'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data)

gamma = 1.5

tr = cy.get_trial_runner(src=src, ana=ana11, flux=cy.hyp.PowerLawFlux(gamma=gamma), mp_cpus=20)



def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials", "bg")

bg = cy.bk.get_all(
        # disk location
        '{}/bg'.format(bg_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)



print("sens, gamma = 1.5")

sens = tr.find_n_sig(
	# ts, threshold
    bg.median(),
    # beta, fraction of trials which should exceed the threshold
    0.9,
    # n_inj step size for initial scan
    n_sig_step=5,
    # this many trials at a time
    batch_size=500,
	# tolerance, as estimated relative error
	tol=.05
	)

print(sens)

gamma = 3.

tr = cy.get_trial_runner(src=src, ana=ana11, flux=cy.hyp.PowerLawFlux(gamma=gamma), mp_cpus=20)

print("sens, gamma = 3.")

sens = tr.find_n_sig(
    # ts, threshold
    bg.median(),
    # beta, fraction of trials which should exceed the threshold
    0.9,
    # n_inj step size for initial scan
    n_sig_step=5,
    # this many trials at a time
    batch_size=500,
    # tolerance, as estimated relative error
    tol=.05
    )

print(sens)

print("Done")
