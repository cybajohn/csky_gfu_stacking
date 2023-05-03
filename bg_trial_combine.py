import numpy as np
import os
import csky as cy
from _paths import PATHS

from IPython import embed
import matplotlib.pyplot as plt
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials_new", "bg_new")

bg = cy.bk.get_all(
        # disk location
        '{}/bg'.format(bg_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)



print(bg)

sigma_5 = bg.isf_nsigma(5)
sigma_3 = bg.isf_nsigma(3)
median = bg.median()

print("bg params: median: ", median, " 3 sigma: ", sigma_3, " 5 sigma: ", sigma_5)

fontsize = 12

fig, ax = plt.subplots()

h = bg.get_hist(bins=50)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8,label=r'$5\sigma$')
ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5,label=r'$3\sigma$')
ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
	item.set_fontsize(fontsize)
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_bg_new.pdf")
plt.clf()

