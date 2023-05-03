"""
Create jobfiles for `bg_trials_time_dep_for_jobs.py`.

##############################################################################
# Used seed range for bg trial jobs: [0, 100000]
##############################################################################
"""

import os
import numpy as np
import csky as cy

from dagman import dagman
from _paths import PATHS

from IPython import embed

from _loader import easy_source_list_loader as src_load


job_creator = dagman.DAGManJobCreator()
job_name = "csky_ehe_transient_stacking"

job_dir = os.path.join(PATHS.jobs, "txs_bg_trials_time_dep")
script = ["~/venvs/py3v4/bin/python3", os.path.join(PATHS.repo, "txs_bg_trial_time_dep_for_jobs.py")]

# cache ana
print("cache ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "txs_bg_time_dep")
# maybe change that to csky methods
if not os.path.isdir(ana_dir):
    os.makedirs(ana_dir)

cy.CONF['mp_cpus'] = 5

remake = True

if len(os.listdir(ana_dir)) == 0 or remake:
        print("caching ana for later")
        ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_uncleaned_data,
        )
        ana11.save(ana_dir)

# load sources

srcs = src_load()

n_srcs = 20

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)

embed()

src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)


src_id = np.array([src_id[0]])
n_srcs = 1

ntrials = 1e6
max_trials_per_job = 1e4
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))


job_args = {"seed": list(np.tile(list(range(100, 100 + njobs)),n_srcs)),
            "id": list(np.tile(["run_{:d}".format(i) for i in range(njobs)],n_srcs)),
            "ntrials": [int(ntrials_per_job) for i in range(njobs*n_srcs)],
            "src_id": list(np.repeat(src_id,njobs))}

#exe = os.path.join("/home", "jkollek", "venvs", "py3v4", "activate.sh")
#                     ,"/bin/bash")
#venvs/py3v4/bin
#job_args = {
#    "rnd_seed": np.arange(10000, 10000 + njobs).astype(int),
#    "ntrials": njobs * ntrials_per_job,
#    "job_id": job_ids,
#    "tw_id": tw_ids,
#    }

exe = "/bin/bash"


job_creator.create_job(job_exe=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, bash_exe=exe, ram="5GB", overwrite=True)


