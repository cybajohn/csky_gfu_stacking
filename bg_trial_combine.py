import numpy as np
import os
import csky as cy
from _paths import PATHS

import matplotlib.pyplot as plt
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials", "bg")

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


fig, ax = plt.subplots()

h = bg.get_hist(bins=50)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/test2.pdf")
plt.clf()

