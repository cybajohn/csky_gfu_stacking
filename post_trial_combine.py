import numpy as np
import os
import csky as cy
from _paths import PATHS
from IPython import embed
from _loader import easy_source_list_loader as src_load

from scipy.stats import chi2
from scipy.stats import norm

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import histlite as hl


bg_dir = os.path.join(PATHS.data, "post_trials_new", "bg", "src")

src_id = os.listdir(bg_dir)

all_p_values = []

for i,src in enumerate(src_id):
    bg = cy.bk.get_all(
        # disk location
        '{}/{}'.format(bg_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate)
    all_p_values.append(np.array(bg))

min_len = len(all_p_values[0])

for item in all_p_values:
    min_len_ = len(item)
    if min_len_ < min_len:
        min_len = min_len_

for i,item in enumerate(all_p_values):
    all_p_values[i] = item[:min_len]

all_p_values = np.array(all_p_values).T

all_min_p_values = np.array([-np.log10(np.amin(1-ps)) for ps in all_p_values])

all_min_p_values_2 = np.array([np.amin(1-ps) for ps in all_p_values])

a,b,c = chi2.fit(all_min_p_values)

def pre_pdf(x):
    return chi2.pdf(x,a,b,c)

hist = plt.hist(all_min_p_values, bins=40, histtype='step', density=True, label=r'{} trials'.format(min_len))
x_space = np.linspace(hist[1][0],hist[1][-1],1000)
plt.plot(x_space, pre_pdf(x_space))
#plt.yscale('log')
plt.ylabel(r'PDF')
plt.xlabel(r'Pre-trial $-\mathrm{log}_{10}(p)$')
plt.legend(loc='best')
plt.savefig('test_plots/pre_trial_p_values.pdf')
plt.clf()

#embed()

def conversion(sigma,a,b,c):
    return 1-(1-norm.cdf(1-chi2.cdf(norm.ppf((sigma-1)/2+1),a,b,c)))*2

def conversion_2(x,a,b,c):
    return 1 - chi2.cdf(x,a,b,c)

def conversion_step_1(x):
    return norm.ppf((x+1)/2)

def conversion_step_2(x):
    return 2*norm.cdf(x) -1

def conversion_3(x,a,b,c):
    return conversion_step_1(conversion_2(conversion_step_2(x),a,b,c))

log_space=np.linspace(0, 6,1000)


a,b,c = chi2.fit(all_min_p_values)

embed()

y = np.quantile(all_min_p_values,1-10**(-log_space))

#ehm...
plt.plot(y,log_space)
#plt.plot(log_space, -np.log10(conversion_2(log_space,a,b,c)))
plt.plot(log_space, log_space, linestyle="--", color="k", lw=1)
plt.ylabel(r'Post-trial $-\mathrm{log}_{10}(p)$')
plt.xlabel(r'Pre-trial $-\mathrm{log}_{10}(p)$')
#plt.legend(loc='best')
plt.savefig('test_plots/post_trial_p_values_1.pdf')
plt.clf()

p_space = np.linspace(0,1,1000)

a,b,c = chi2.fit(all_min_p_values_2)

plt.plot(p_space, conversion_2(p_space,a,b,c))
#plt.plot(sigma_space, sigma_space, linestyle="--")
#plt.xlim(0,5)
#plt.ylim(0,5)
plt.ylabel(r'CDF')
plt.xlabel(r'Pre-trial $p$ value')
#plt.legend(loc='best')
plt.savefig('test_plots/post_trial_p_values_2.pdf')
plt.clf()

sigma_space = np.linspace(0,5,1000)

plt.plot(sigma_space, conversion_3(sigma_space,a,b,c))
plt.plot(sigma_space, sigma_space, linestyle="--")
#plt.xlim(0,5)
#plt.ylim(0,5)
plt.ylabel(r'Post-trial $\sigma$')
plt.xlabel(r'Pre-trial $\sigma$ value')
#plt.legend(loc='best')
plt.savefig('test_plots/post_trial_p_values_3.pdf')
plt.clf()




