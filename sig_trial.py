"""
this is time-integrated for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt

from _loader import source_list_loader
import csky as cy
import histlite as hl

# load sources

names = source_list_loader()

names = names[1:-1]

src_ra = []
src_dec = []

for name in names:
        srcs = source_list_loader(name)
        srcs_ra = [src["ra"] for src in srcs[name]]
        srcs_dec = [src["dec"] for src in srcs[name]]
        src_ra.extend(srcs_ra)
        src_dec.extend(srcs_dec)

print(srcs_ra)
print(srcs_dec)

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
ana_dir = cy.utils.ensure_dir('/data/user/jkollek/csky_cache/ana/')
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-003-p03', cy.selections.PSDataSpecs.ps_2011,
                                            'version-003-p03', cy.selections.PSDataSpecs.IC86_2012_2014,
                                         dir=ana_dir)

# get trial runner
tr = cy.get_trial_runner(src=src, ana=ana11, mp_cpus=20)
# run signal trials

N = 1000
n_sig = 10
seed = 1
trials = tr.get_many_fits(N, n_sig, poisson=True, seed=seed)

print(trials)
print("Done")




