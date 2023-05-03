import numpy as np
import matplotlib.pyplot as plt
from _paths import PATHS
import argparse
import os

from _loader import source_list_loader
import csky as cy



# load sources

names = source_list_loader()

names = names[2:-1]

src_ra = []
src_dec = []

for name in names:
        srcs = source_list_loader(name)
        srcs_ra = [src["ra"] for src in srcs[name]]
        srcs_dec = [src["dec"] for src in srcs[name]]
        src_ra.extend(srcs_ra)
        src_dec.extend(srcs_dec)

# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)




ana_dir = os.path.join(PATHS.data, "ana_cache", "sig")
ana11 = cy.get_analysis(cy.selections.repo,
#                                            'version-003-p03', cy.selections.PSDataSpecs.IC79,
#                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
#                                            'version-003-p03', cy.selections.PSDataSpecs.IC86v3_2015,
                                             dir=ana_dir)

# get trial runner
tr = cy.get_trial_runner(src=src, ana=ana11,flux=cy.hyp.PowerLawFlux(3))

# params

gamma = 3.0
n_sig = 10.0
rnd_seed = 100
ntrials = 1000


#adjust sig inj flux
for j in range(len(tr.sig_injs)):
	for i in range(len(tr.sig_injs[j].flux)):
		print('old: ',tr.sig_injs[j].flux[i])
		tr.sig_injs[j].flux[i] = cy.hyp.PowerLawFlux(gamma=gamma)
		print('new: ',tr.sig_injs[j].flux[i])

print("gamma: ",gamma," flux: ",tr.sig_injs[0].flux)

trials = tr.get_many_fits(ntrials, n_sig, poisson=True, seed=rnd_seed, logging=True, mp_cpus=10)

from IPython import embed
embed()



