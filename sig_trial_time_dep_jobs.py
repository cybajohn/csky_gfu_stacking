"""
Create jobfiles for `sig_trial_time_dep_for_jobs.py`.

##############################################################################
# Used seed range for sig trial jobs: [100, 100 + njobs] #not really
##############################################################################
"""

import os
import numpy as np
import csky as cy

from IPython import embed

from dagman import dagman
from _paths import PATHS
from _loader import easy_source_list_loader as src_load


job_creator = dagman.DAGManJobCreator()
job_name = "csky_ehe_transient_stacking"

job_dir = os.path.join(PATHS.jobs, "sig_trials_time_dep_t0_dt_gamma_ran_new")
script = ["~/venvs/py3v4/bin/python3", os.path.join(PATHS.repo, "sig_trial_time_dep_for_jobs.py")]

# cache ana
print("cache ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_time_dep_t0_dt_gamma_ran_new")
# maybe change that to csky methods
if not os.path.isdir(ana_dir):
    os.makedirs(ana_dir)

cy.CONF['mp_cpus'] = 5

remake = True

if len(os.listdir(ana_dir)) == 0 or remake:
	print("caching ana for later")
	ana11 = cy.get_analysis(cy.selections.repo,
					    'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
	)
	ana11.save(ana_dir)
 
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

#embed()

src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

embed()

print("setting up job args")

ntrials = 1e6
max_trials_per_job = 1e4
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))

#only gamma = 2 to simplify, 40 srcs are already plenty
"""
gamma_min = 1
gamma_max = 4
gamma_steps = 7
"""

gamma = [2.0] # only 2 to test
n_gamma = len(gamma)
#gamma = list(np.round(np.linspace(gamma_min,gamma_max,gamma_steps),decimals=1))

njobs_total = int(njobs * n_srcs * n_gamma)

print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles".format(int(njobs_total)))


job_ids = np.arange(0,njobs_total).astype(int)
n_sig_min = 0
n_sig_max = 10
n_sig_steps = 10
n_sig = list(np.round(np.linspace(n_sig_min,n_sig_max,n_sig_steps),decimals=1))

more_n_sig = list(np.round(np.linspace(12,30,10),decimals=1))
n_sig.extend(more_n_sig)


n_sig_steps+= 10

n_jobs_per_signal = int(njobs/n_sig_steps)


print("{} jobs per signal".format(n_jobs_per_signal))

job_args = {"seed": list(np.repeat(np.tile(np.tile(list(np.array(range(100, 100 + n_jobs_per_signal))*ntrials_per_job),n_srcs),n_sig_steps),n_gamma)),
            "id": ["run_{:d}".format(i) for i in range(njobs_total)],
            "ntrials": [int(ntrials_per_job) for i in range(njobs_total)],
	    "n_sig": list(np.repeat(np.repeat(np.tile(n_sig,n_srcs),n_jobs_per_signal),n_gamma)),
	    "src_id": list(np.repeat(np.repeat(src_id,njobs),n_gamma)),
        "gamma": list(np.tile(np.tile(gamma,njobs),n_srcs))}
print("check job_args")
embed()

exe = "/bin/bash"

#create data outpath here
outpath = os.path.join(PATHS.data, "sig_trials_time_dep_t0_dt_gamma_ran_new")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

job_creator.create_job(job_exe=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, bash_exe=exe, ram="6GB", overwrite=True)



