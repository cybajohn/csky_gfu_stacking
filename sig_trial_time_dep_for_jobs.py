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
parser.add_argument("--src_id", type=int)
parser.add_argument("--gamma", type=float)
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
n_sig = args.n_sig
gamma = args.gamma
src_id = args.src_id


outpath = os.path.join(PATHS.data, "sig_trials_time_dep_t0_dt_gamma_ran_new")

trials_dir = outpath

sig_dir = cy.utils.ensure_dir('{}/sig'.format(trials_dir))

# load mc, data

ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_time_dep_t0_dt_gamma_ran_new")


ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

src = cy.utils.Sources(ra=srcs[src_id]["ra"], dec=srcs[src_id]["dec"])


# make single trial_runners for every random emission window
sig = []
for i in range(0,ntrials):
    conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': srcs[src_id]['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=srcs[src_id]['mjd'], dt=np.random.uniform(1/24.,200), flux=cy.hyp.PowerLawFlux(gamma)),
    }
    tr = cy.get_trial_runner(conf=conf_box, ana=ana11, src=src, dt_max=200)
    fit = tr.get_many_fits(1,n_sig, poisson=True, TRUTH=False, seed=rnd_seed+i, logging=False, _fmin_method='minuit')
    sig.append(fit)

trials = cy.dists.utils.Arrays.concatenate(sig)


"""
#old code
conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': srcs[src_id]['mjd']}, 
    'seeder': cy.seeding.UTFSeeder(), 
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=srcs[src_id]['mjd'], dt=[1/24.,200], flux=cy.hyp.PowerLawFlux(gamma)),
     }

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)

trials = tr_box.get_many_fits(ntrials, n_sig, poisson=True, seed=rnd_seed, logging=False, _fmin_method='minuit')
"""

# save to disk
directory = cy.utils.ensure_dir('{}/time_dep/gamma/{}/src/{}/sig/{}'.format(sig_dir,gamma,src_id,n_sig))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
# notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
np.save(filename, trials.as_array)

print("Done")

