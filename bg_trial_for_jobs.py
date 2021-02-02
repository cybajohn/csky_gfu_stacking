import numpy as np
import matplotlib.pyplot as plt
from _paths import PATHS
import argparse
import os

from _loader import source_list_loader
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
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id



# load sources

names = source_list_loader()

name = names[1]

srcs = source_list_loader(name)
print(srcs)
srcs_ra = [src["ra"] for src in srcs[name]]
print(srcs_ra)
srcs_dec = [src["dec"] for src in srcs[name]]
print(srcs_dec)
# convert sources to csky_style

src = cy.utils.Sources(ra=srcs_ra, dec=srcs_dec)

print(src)

outpath = os.path.join(PATHS.data, "bg_trials")
if not os.path.isdir(outpath):
    os.makedirs(outpath)


#trials_dir = cy.utils.ensure_dir(outpath + './trials/IC86_2011')
trials_dir = outpath

bg_dir = cy.utils.ensure_dir('{}/bg'.format(trials_dir))

# load bg

#cy.selections.DataSpec._version = 'version-003-p03'
ana_dir = cy.utils.ensure_dir(PATHS.data+'/csky_cache/ana/')
ana11 = cy.get_analysis(cy.selections.repo, 'version-003-p03', cy.selections.PSDataSpecs.ps_2011, dir=ana_dir)

tr = cy.get_trial_runner(src=src, ana=ana11)


#bg = cy.dists.Chi2TSD(tr.get_many_fits(1000, seed=1))

trials = tr.get_many_fits(ntrials, seed=rnd_seed, logging=False)
# save to disk
directory = cy.utils.ensure_dir('{}/bg'.format(bg_dir))
filename = '{}/trials__N_{:06d}_seed_{:04d}_job_{}.npy'.format(directory, ntrials, rnd_seed, job_id)
print('->', filename)
# notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
np.save(filename, trials.as_array)

print("Done")

print(filename)
