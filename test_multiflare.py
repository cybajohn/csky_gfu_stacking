"""
to test multiflare
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from _paths import PATHS
from IPython import embed


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


n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)


src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

src_time = [_src['mjd'] for _src in srcs]

test_src = srcs[src_id[9]]

# convert sources to csky_style

src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'])

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=test_src['mjd'], flux=cy.hyp.PowerLawFlux(2.0)),
     }


N_trials = 3
n_sig = 10
sig = []

timer = cy.timing.Timer()
time = timer.time

with time('tr_runner for every single trial'):
    for i in range(0,N_trials):
        #print(i,"/",N_trials)
        conf_box = {
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': test_src['mjd']},
        'seeder': cy.seeding.UTFSeeder(),
        'sig' : 'tw',
        'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=np.random.uniform(1/24.,200), flux=cy.hyp.PowerLawFlux(2.0)),
        }
        tr_1 = cy.get_trial_runner(conf=conf_box, ana=ana11, src=src, dt_max=200)
        this_thing = tr_1.get_many_fits(1,n_sig, poisson=True, TRUTH=False, logging=False, seed=i, _fmin_method='minuit')
        sig.append(this_thing) 

sig = cy.dists.utils.Arrays.concatenate(sig)
embed()

with time('multi_trial_runner'):
    tr = cy.get_multiflare_trial_runner(conf=conf_box, ana=ana11, src=src, dt_max=200)
    
    fits = []
    for i in range(0,N_trials):
        print(i,"/",N_trials)
        n_evts = np.random.poisson(n_sig)
        t_max = np.random.uniform(1/24.,200,1)
        src_times = np.random.uniform(0, t_max, n_evts) #These are the times of our individual events. For a different flare duration, you'll have to adjust t_min and t_max
        csky_ind = 0 #The index corresponding to which source in your cy.sources object you want to inject these events on
        inj_flares = [[src_times, csky_ind]]
        this_trial = tr.get_one_trial(injflares=inj_flares, poisson=False, TRUTH=False, seed=i)
        this_fit = tr.get_one_fit_from_trial(this_trial, flat=False)
        fits.append(this_fit)
embed()

