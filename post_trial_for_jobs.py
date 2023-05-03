import numpy as np
import matplotlib.pyplot as plt
from _paths import PATHS
import argparse
import os

from _loader import easy_source_list_loader as src_load

from scipy.integrate import quad as integrate

import csky as cy
import histlite as hl

import sys

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

print("hello there")
print("Args are:")
for arg in sys.argv[1:]:
    print(arg)

parser = argparse.ArgumentParser(description="ehe_stacking")
parser.add_argument("--seed", type=int)
parser.add_argument("--id", type=str)
parser.add_argument("--ntrials", type=int)
parser.add_argument("--src_id", type=int)
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
src_id = args.src_id



outpath = os.path.join(PATHS.data, "post_trials_new")

trials_dir = outpath

bg_dir = cy.utils.ensure_dir('{}/bg'.format(trials_dir))

# load bg

ana_dir = os.path.join(PATHS.data, "ana_cache", "bg_time_dep_t0_dt_ran_new")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

src = cy.utils.Sources(ra=srcs[src_id]["ra"], dec=srcs[src_id]["dec"])


old_bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran_new", "bg", "src")


all_p_values = []

bg = cy.bk.get_all(
    # disk location
    '{}/{}'.format(old_bg_dir,src_id),
    # filename pattern
    'trials*npy',
    # how to combine items within each directory
    merge=np.concatenate,
    # what to do with items after merge
    post_convert=ndarray_to_Chi2TSD)

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': srcs[int(src_id)]['mjd']},
}
src = cy.utils.Sources(ra=srcs[int(src_id)]['ra'], dec=srcs[int(src_id)]['dec'])
tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)

bg_for_p = tr_box.get_many_fits(ntrials, seed=rnd_seed, logging=False, _fmin_method='minuit')
p_values = []
for i,ts in enumerate(bg_for_p.ts):
    p_values.append(bg.cdf(ts))
    #p_values.append(integrate(lambda x: bg.pdf(x), ts, np.inf)[0])
p_values = np.array(p_values)



# save to disk
directory = cy.utils.ensure_dir('{}/src/{}'.format(bg_dir,src_id))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
np.save(filename, p_values)

print("Done")

print(filename)


