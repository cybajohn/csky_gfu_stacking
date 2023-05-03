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


# convert sources to csky_style

src = cy.utils.Sources(ra=src_ra, dec=src_dec)

print(src)

# load bg

ana11 = cy.get_analysis(cy.selections.repo, 
					    'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data)

tr = cy.get_trial_runner(src=src, ana=ana11, mp_cpus=20)


bg = cy.dists.Chi2TSD(tr.get_many_fits(1000, seed=1))

print(bg)
print(bg.description)

# convert bg to csky_style

# set up csky options? (going with defaults first)

# do trials (maybe 100?)

# maybe plot trials?

fig, ax = plt.subplots()

h = bg.get_hist(bins=15)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/test1.pdf")
plt.clf()
