"""
Create jobfiles for `post_trial_for_jobs.py`.
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

job_dir = os.path.join(PATHS.jobs, "post_trials_new")
script = ["~/venvs/py3v4/bin/python3", os.path.join(PATHS.repo, "post_trial_for_jobs.py")]

outpath_data = os.path.join(PATHS.data, "post_trials_new")
if not os.path.isdir(outpath_data):
    os.makedirs(outpath_data)


# cache ana
print("cache ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "bg_time_dep_t0_dt_ran_new")
# maybe change that to csky methods
if not os.path.isdir(ana_dir):
    os.makedirs(ana_dir)

cy.CONF['mp_cpus'] = 5

ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)

# load sources

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs_all = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

if len(srcs_all) < len(srcs):
    print("Number of possible sources reduced ({} -> {}) due to analysis time frame".format(len(srcs),len(srcs_all)))
    srcs = srcs_all

n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)


src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)


ntrials = 1e7/2.
max_trials_per_job = 1e5/2.
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))


njobs_total = int(njobs * n_srcs)

print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles".format(int(njobs_total)))


job_args = {"seed": list(np.tile(list(np.array(range(100, 100 + njobs))*ntrials_per_job),n_srcs)),
            "id": ["run_{:d}".format(i) for i in range(njobs_total)],
            "ntrials": [int(ntrials_per_job) for i in range(njobs_total)],
            "src_id": list(np.repeat(src_id,njobs))}
print("check job_args")
embed()

exe = "/bin/bash"


job_creator.create_job(job_exe=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, bash_exe=exe, ram="7GB", overwrite=True)


