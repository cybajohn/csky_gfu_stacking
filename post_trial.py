# p-values
import numpy as np
import matplotlib.pyplot as plt
import os
from _paths import PATHS
from IPython import embed

from scipy.stats import chi2
from tqdm import tqdm

from scipy.integrate import quad as integrate

from _loader import easy_source_list_loader as src_load
import csky as cy
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

ana_dir="saved_models_for_test/sig_trial_time_dep"
cy.CONF['mp_cpus'] = 10
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)

ana11.save(ana_dir)

t_max = ana11.mjd_max
t_min = ana11.mjd_min

srcs = src_load()

# Check if sources are inside the analysis time frame
srcs = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran", "bg", "src")

src_id = os.listdir(bg_dir)

all_p_values = []

for i,srcid in enumerate(src_id):
    bg = cy.bk.get_all(
        # disk location
        '{}/{}'.format(bg_dir,srcid),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)
    conf_box = {
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': srcs[int(srcid)]['mjd']},
     }
    src = cy.utils.Sources(ra=srcs[int(srcid)]['ra'], dec=srcs[int(srcid)]['dec'])
    tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)
    
    bg_for_p = tr_box.get_many_fits(100, seed=1, _fmin_method='minuit', mp_cpus=10)
    p_values = []
    for i,ts in tqdm(enumerate(bg_for_p.ts)):
        p_values.append(1-bg.cdf(ts))
    all_p_values.append(np.array(p_values))

all_p_values = np.array(all_p_values).T

all_min_p_values = np.array([-np.log10(np.amin(ps)) for ps in all_p_values])
all_max_p_values = np.array([-np.log10(np.amax(ps)) for ps in all_p_values])

plt.hist(all_min_p_values, bins=20, histtype='step', density=True)
plt.yscale('log')
plt.savefig('test_plots/p_value_test.pdf')
plt.clf()

plt.hist(all_max_p_values, density=True)
plt.yscale('log')
plt.savefig('test_plots/p_value_test2.pdf')
plt.clf()

