"""
Create jobfiles for `sig_trial_for_jobs.py`.

##############################################################################
# Used seed range for sig trial jobs: [100, 100 + njobs] #not really
##############################################################################
"""

import os
import numpy as np
import csky as cy

from dagman import dagman
from _paths import PATHS


job_creator = dagman.DAGManJobCreator()
job_name = "csky_ehe_transient_stacking"

job_dir = os.path.join(PATHS.jobs, "sig_trials_new_2")
script = ["~/venvs/py3v4/bin/python3", os.path.join(PATHS.repo, "sig_trial_for_jobs.py")]

# cache ana
print("cache ana")
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_new")
# maybe change that to csky methods
if not os.path.isdir(ana_dir):
    os.makedirs(ana_dir)

cy.CONF['mp_cpus'] = 5

remake = True

if len(os.listdir(ana_dir)) == 0 or remake:
	print("caching ana for later")
	"""
	ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC79,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86v3_2015, 
	)
	"""
	ana11 = cy.get_analysis(cy.selections.repo,
					    'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
	)
	ana11.save(ana_dir)
 
#ana11.save(ana_dir)

print("setting up job args")

ntrials = 1e6
max_trials_per_job = 0.5*1e4
njobs = int(np.ceil(ntrials/float(max_trials_per_job)))
ntrials_per_job = int(ntrials/float(njobs))

gamma_min = 1.5
gamma_max = 3.
gamma_steps = 7

gamma = list(np.round(np.linspace(gamma_min,gamma_max,gamma_steps),decimals=4))

njobs_total = int(njobs * gamma_steps)

print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles".format(int(njobs_total)))


job_ids = np.arange(0,njobs_total).astype(int)
n_sig_min = 0
n_sig_max = 30
n_sig_steps = 5
n_sig = list(np.round(np.linspace(n_sig_min,n_sig_max,n_sig_steps),decimals=1))

more_n_sig = list(np.round(np.linspace(36,120,5),decimals=1))
even_more_n_sig = list(np.round(np.linspace(141,620,10),decimals=1))

n_sig.extend(more_n_sig)
n_sig.extend(even_more_n_sig)

n_sig_steps+=15

n_jobs_per_signal = int(njobs/n_sig_steps)


print("{} jobs per signal".format(n_jobs_per_signal))

job_args = {"seed": list(np.tile(np.tile(list(range(100, 100 + n_jobs_per_signal)),gamma_steps),n_sig_steps)),
            "id": ["run_{:d}".format(i) for i in range(njobs_total)],
            "ntrials": [int(ntrials_per_job) for i in range(njobs_total)],
	    "n_sig": list(np.repeat(np.tile(n_sig,gamma_steps),n_jobs_per_signal)),
	    "gamma": list(np.repeat(gamma,njobs))}


exe = "/bin/bash"

job_creator.create_job(job_exe=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, bash_exe=exe, ram="6GB", overwrite=True)



