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
parser.add_argument("--src_id", type=int)
#parser.add_argument("--time_window_length", type=float)
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
src_id = args.src_id
#time_window_length = args.time_window_length

# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]
src_mjd = [src["mjd"] for src in srcs]


outpath = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran_new")
if not os.path.isdir(outpath):
    os.makedirs(outpath)


#trials_dir = cy.utils.ensure_dir(outpath + './trials/IC86_2011')
trials_dir = outpath

bg_dir = cy.utils.ensure_dir('{}/bg'.format(trials_dir))

# load bg

#cy.selections.DataSpec._version = 'version-003-p03'
#ana_dir = cy.utils.ensure_dir(PATHS.data+'/csky_cache/ana/')
ana_dir = os.path.join(PATHS.data, "ana_cache", "bg_time_dep_t0_dt_ran_new")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]


"""
Configuration = {'ana': ana11,
                 'space': "ps",
                 'time': "transient",
                 'sig': 'transient'}

cy.CONF.update(Configuration)

mjd = np.array(src_mjd[src_id]) - np.array(time_window_length)/2 # Start of box time window
t100 = np.array(time_window_length) # Width of box time window, in days
src = cy.utils.Sources(ra=src_ra[src_id], dec=src_dec[src_id], deg=False, mjd=mjd, t_100=t100, sigma_t=1*[0])

trtr = cy.get_trial_runner(src=src)

"""

src = cy.utils.Sources(ra=srcs[src_id]["ra"], dec=srcs[src_id]["dec"])

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': srcs[src_id]['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
     }

trtr = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)



trials = trtr.get_many_fits(ntrials, seed=rnd_seed, logging=False, _fmin_method='minuit')

# save to disk
directory = cy.utils.ensure_dir('{}/src/{}'.format(bg_dir,src_id))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
# notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
np.save(filename, trials.as_array)

print("Done")

print(filename)


