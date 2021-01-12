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

srcs = source_list_loader(names[2])
print(srcs)
srcs_ra = [src["ra"] for src in srcs[names[2]]]
print(srcs_ra)
srcs_dec = [src["dec"] for src in srcs[names[2]]]
print(srcs_dec)
# convert sources to csky_style

src = cy.utils.Sources(ra=srcs_ra, dec=srcs_dec)

print(src)

# load bg

cy.selections.DataSpec._version = 'version-003-p03'
ana_dir = cy.utils.ensure_dir('/data/user/jkollek/csky_cache/ana/')
ana11 = cy.get_analysis(cy.selections.repo, cy.selections.PSDataSpecs.ps_2011, dir=ana_dir)

tr = cy.get_trial_runner(src=src, ana=ana11, sindec_bandwidth=np.radians(.1), mp_cpus=20)


bg = cy.dists.Chi2TSD(tr.get_many_fits(100, seed=1))

print(bg)
print(bg.description)

# convert bg to csky_style

# set up csky options? (going with defaults first)

# do trials (maybe 100?)

# maybe plot trials?

fig, ax = plt.subplots()

h = bg.get_hist(bins=30)
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
