"""
this is time-integrated for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from _loader import source_list_loader
import csky as cy
import histlite as hl

# load sources
print("load source(s)")
names = source_list_loader()

names = names[1:-1]

src_ra = []
src_dec = []

for name in names:
        srcs = source_list_loader(name)
        srcs_ra = [src["ra"] for src in srcs[name]]
        srcs_dec = [src["dec"] for src in srcs[name]]
        src_ra.extend(srcs_ra)
        src_dec.extend(srcs_dec)



# convert sources to csky_style
# only 1 source
source_n = 0 # take first source
src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)

# load data, mc
print("load data, mc")
path = 'saved_models_for_test/single_source'
ana_dir = cy.utils.ensure_dir(path)
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                         dir=ana_dir)
# if this is the first time, save this ana to save time later on
ana_count = 2
if len(os.listdir(path)) < ana_count:
	print("caching ana for later")
	ana11.save(ana_dir)

# set up trial runner
print("set up trial runner")
tr = cy.get_trial_runner(src=src, ana=ana11, mp_cpus=10)

# run bg trials
print("do bg trials")
n_trials = 1000
seed = 1
bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials, seed=seed))

# ensure signal does something
print("is the signal injector doing something? -:")
print(tr.get_one_fit(n_sig=100, seed=1))

# give sens
print("start find_n_sig")
sens = tr.find_n_sig(
        bg.median(), 0.95,
        n_sig_step=2,
        first_batch_size=100,
        batch_size=200,
        # 10% tolerance -- let's get an estimate quick!
        tol=.05,
        # number of signal signal strengths (default 6, i'm really tryina rush here)
        n_batches=10
    )




