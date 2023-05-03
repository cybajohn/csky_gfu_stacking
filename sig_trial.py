"""
this is time-integrated for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt

from _loader import easy_source_list_loader as src_load

import csky as cy
import histlite as hl

# load sources

srcs = src_load()

src_ra = [src["ra"] for src in srcs]
src_dec = [src["dec"] for src in srcs]

#srcs = source_list_loader(name)
#print(srcs)
#srcs_ra = [src["ra"] for src in srcs[name]]
#print(srcs_ra)
#srcs_dec = [src["dec"] for src in srcs[name]]
#print(srcs_dec)



# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)


# get data and mc
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                         )

# get trial runner
tr = cy.get_trial_runner(src=src, ana=ana11, mp_cpus=20)
# run signal trials

N = 1000
n_sig = 10
seed = 1
trials = tr.get_many_fits(N, n_sig, poisson=True, seed=seed)

print(trials)
print("Done")




