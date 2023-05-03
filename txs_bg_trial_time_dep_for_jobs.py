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
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
src_id = args.src_id


# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]

src = cy.utils.Sources(ra=src_ra[src_id], dec=src_dec[src_id])

print(src)

outpath = os.path.join(PATHS.data, "txs_bg_trials_time_dep")
if not os.path.isdir(outpath):
    os.makedirs(outpath)


#trials_dir = cy.utils.ensure_dir(outpath + './trials/IC86_2011')
trials_dir = outpath

bg_dir = cy.utils.ensure_dir('{}/bg'.format(trials_dir))

# load bg

#cy.selections.DataSpec._version = 'version-003-p03'
#ana_dir = cy.utils.ensure_dir(PATHS.data+'/csky_cache/ana/')
ana_dir = os.path.join(PATHS.data, "ana_cache", "txs_bg_time_dep")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_uncleaned_data,
                                            dir=ana_dir)

conf_box = {
    'time': 'utf',
    'box': True,
    'seeder': cy.seeding.UTFSeeder(),
    'dt_max': 400,
    }



tr = cy.get_trial_runner(conf_box, src=src, ana=ana11)

trials = tr.get_many_fits(ntrials, seed=rnd_seed, logging=False)
# save to disk
directory = cy.utils.ensure_dir('{}/src/{}'.format(bg_dir,src_id))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
# notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
np.save(filename, trials.as_array)

print("Done")

print(filename)


