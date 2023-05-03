import numpy as np
import matplotlib.pyplot as plt
from _paths import PATHS
import argparse
import os

from _loader import easy_source_list_loader as src_load
import csky as cy
import histlite as hl

import sys


print("hello there")
print("Args are:")
for arg in sys.argv[1:]:
    print(arg)

parser = argparse.ArgumentParser(description="ehe_stacking")
parser.add_argument("--seed", type=int)
parser.add_argument("--id", type=str)
parser.add_argument("--ntrials", type=int)
parser.add_argument("--n_sig", type=float)
parser.add_argument("--gamma", type=float)
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
n_sig = args.n_sig
gamma = args.gamma



# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]

# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)

outpath = os.path.join(PATHS.data, "sig_trials_new_2")
if not os.path.isdir(outpath):
    os.makedirs(outpath)


trials_dir = outpath

sig_dir = cy.utils.ensure_dir('{}/sig_new'.format(trials_dir))

# load mc, data

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


# get trial runner
tr = cy.get_trial_runner(src=src, ana=ana11,flux=cy.hyp.PowerLawFlux(gamma=gamma))
"""
#adjust sig inj flux
for j in range(len(tr.sig_injs)):
	for i in range(len(tr.sig_injs[j].flux)):
		tr.sig_injs[j].flux[i] = cy.hyp.PowerLawFlux(gamma=gamma)
"""
print("gamma: ",gamma," flux: ",tr.sig_injs[0].flux)


trials = tr.get_many_fits(ntrials, n_sig, poisson=True, seed=rnd_seed, logging=False)
# save to disk
directory = cy.utils.ensure_dir('{}/for_gamma_3/gamma/{}/sig/{}'.format(sig_dir,gamma,n_sig))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
# notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
np.save(filename, trials.as_array)

print("Done")

