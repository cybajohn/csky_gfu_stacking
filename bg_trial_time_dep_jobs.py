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

job_dir = os.path.join(PATHS.jobs, "bg_trials_time_dep_t0_dt_ran_new")
script = ["~/venvs/py3v4/bin/python3", os.path.join(PATHS.repo, "bg_trial_time_dep_for_jobs.py")]

# cache ana
print("cache ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "bg_time_dep_t0_dt_ran_new")
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

ntrials = 1e6
max_trials_per_job = 1e4
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))


njobs_total = int(njobs * n_srcs)

print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles".format(int(njobs_total)))


#job_ids = np.arange(0,njobs_total).astype(int)
#n_sig_min = 0
#n_sig_max = 16
#n_sig_steps = 10
#n_sig = list(np.round(np.linspace(n_sig_min,n_sig_max,n_sig_steps),decimals=1))

#more_n_sig = list(np.round(np.linspace(18,100,10),decimals=1))
#even_more_n_sig = list(np.round(np.linspace(120,400,5),decimals=1))

#n_sig.extend(more_n_sig)
#n_sig.extend(even_more_n_sig)


#n_sig_steps+=15

time_window_lengths = [1./24,1.,10.,100.,200.] # in days
n_time_windows = len(time_window_lengths)

n_jobs_per_time_window = int(njobs/n_time_windows)


"""
ntrials = 1e6
max_trials_per_job = 1e4/2
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))
"""
"""
job_args = {"seed": list(np.repeat(np.tile(list(range(100, 100 + n_jobs_per_time_window)),n_srcs), n_time_windows)),
            "id": list(np.repeat(np.tile(["run_{:d}".format(i) for i in range(njobs)],n_srcs), n_time_windows)),
            "ntrials": [int(ntrials_per_job) for i in range(njobs*n_srcs*n_time_windows)],
            "src_id": list(np.repeat(np.repeat(src_id,njobs), n_time_windows)),
            "time_window_length": list(np.tile(np.repeat(time_window_lengths,njobs), n_srcs))}
"""
"""
job_args = {"seed": list(np.tile(np.tile(list(range(100, 100 + n_jobs_per_time_window)),n_srcs),n_time_windows)),
            "id": ["run_{:d}".format(i) for i in range(njobs_total)],
            "ntrials": [int(ntrials_per_job) for i in range(njobs_total)],
        "time_window_length": list(np.repeat(np.tile(time_window_lengths,n_srcs),n_jobs_per_time_window)),
        "src_id": list(np.repeat(src_id,njobs))}
"""
job_args = {"seed": list(np.tile(list(range(100, 100 + njobs)),n_srcs)),
            "id": ["run_{:d}".format(i) for i in range(njobs_total)],
            "ntrials": [int(ntrials_per_job) for i in range(njobs_total)],
            "src_id": list(np.repeat(src_id,njobs))}


#embed()

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


