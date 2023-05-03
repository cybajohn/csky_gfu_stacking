"""
this is time-dependent for practice purposes
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from _paths import PATHS
from IPython import embed


from _loader import easy_source_list_loader as src_load
import csky as cy
import histlite as hl

# load sources

# load sources

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


n_srcs = 20

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)


src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

src_time = [_src['mjd'] for _src in srcs]

test_src = srcs[src_id[10]] #9

# convert sources to csky_style

src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'])

bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0", "bg", "src", str(src_id[9]))

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=test_src['mjd'], dt={"min":[1/24.],"max":[200]}, flux=cy.hyp.PowerLawFlux(2.0)),
     }

tr_uncut = cy.get_trial_runner(conf=conf_box,src=src, ana=ana11, dt_max=200, _fmin_method='minuit')

print("check tr_uncut")
embed()

L_uncut = tr_uncut.get_one_llh(n_sig=10, poisson=True, seed=1)

scan_ts,mesh = L_uncut.scan_ts(ns=np.linspace(0,20,50), gamma=np.linspace(1, 4, 31), t0=test_src["mjd"], dt=200)
embed()

x = np.linspace(0,20,50)
y = np.linspace(1,4,31)

xm,ym = np.meshgrid(x,y)

plt.pcolormesh(mesh[0].reshape(50,31), mesh[1].reshape(50,31), scan_ts.reshape(50,31))
plt.xlabel(r"signal parameter $n_s$")
plt.ylabel(r"spectral index $\gamma$")
plt.colorbar()
plt.tight_layout()
plt.savefig("test_plots/llh_scan.pdf")

plt.clf()

# concatenate background and signal S/B values

def SB(_gamma):
    SB_space = np.concatenate([cy.inspect.get_space_eval(L_uncut, -1, i)()[0] for i in (0, 1)])
    SB_energy = np.concatenate([cy.inspect.get_energy_eval(L_uncut, -1, i)(gamma=_gamma)[0] for i in (0, 1)])
    return SB_space * SB_energy

@np.vectorize
def ts(gam,ns):
    return 2 * np.sum(np.log1p(ns * (SB(gam) - 1) / SB(gam).size))

gamma_values = np.linspace(0,4,100)
ns_values = [0,2,4,6,7,8,10]
for item in ns_values:
    plt.plot(gamma_values,ts(gamma_values,item),label="ts for ns={}".format(item))
plt.xlabel("spectral index")
plt.ylabel("ts")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/llh_test.pdf")

plt.clf()


print("L_uncut things to inspect the likelyhood, ts(gam,ns)")
embed()

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200, _fmin_method='minuit')

#tr_box.fitter_args["seeder"].gammas = np.array([2]) #weird...

n_sig = 10 #10

fits = []
evts = []

for i in range(0,100):
    trial = tr_box.get_one_trial(n_sig, poisson=True, seed=i)
    fit = tr_box.get_one_fit_from_trial(trial,_fmin_method='minuit')
    a, b = tr_box.get_one_llh_from_trial(trial).scan_ts(np.array([fit[1]]),gamma=np.array([1,2,3,4]),t0=test_src["mjd"],dt=fit[2])
    comp_2 = int(b[1][0][np.argmax(a)])
    comp_1 = fit[3]
    print("are these the same? ",comp_1,comp_2)
    try:
        evts.append(len(trial.evss[0][1]))
    except:
        evts.append(0)
    fits.append(fit)

print("check trial")
embed()

events = np.array(fits).T[0]

plt.hist(evts,bins=10)
plt.xlabel("events")
plt.ylabel("number of trials")
plt.tight_layout()
plt.savefig("test_plots/event_test.pdf")

plt.clf()

sig = tr_box.get_many_fits(100, n_sig, poisson=True, seed=1, _fmin_method='minuit')
sig = cy.dists.Chi2TSD(sig)

print("test fits, test sig, copy them...")
embed()

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
    #'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=200, flux=cy.hyp.PowerLawFlux(2.0)),
     }

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)

n_sig = 10

sig_1 = tr_box.get_many_fits(100, n_sig, poisson=True, seed=1, mp_cpus=10, _fmin_method='minuit')

sig_1 = cy.dists.Chi2TSD(sig_1)

bg_1 = tr_box.get_many_fits(100,seed=1, mp_cpus=10)

print("test sig_w and sig_1 and bg_1")
embed()
conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=[0,200], flux=cy.hyp.PowerLawFlux(2.0)),
     }

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)

n_sig = 10

sig_new = tr_box.get_many_fits(100, n_sig, poisson=True, seed=1, mp_cpus=10)

sig_new = cy.dists.Chi2TSD(sig_new)
print("test new csky")

embed()
print(sig)
print("new method:")
n_trials = 100
sig_new = []
rnd_seed = 1
for i in range(n_trials):
    conf_box = {
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': test_src['mjd']},
        'seeder': cy.seeding.UTFSeeder(),
        'sig' : 'tw',
        'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=np.random.uniform(0,200,1), flux=cy.hyp.PowerLawFlux(2.0)),
        }
    
    tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)
    
    this_trial = tr_box.get_one_trial(n_sig, poisson=True, seed=rnd_seed+i)
    this_fit = tr_box.get_one_fit_from_trial(this_trial)
    sig_new.append(this_fit)
embed()

conf_box = {
    'time': 'utf',
    'box': True,
    'fitter_args': {'t0': test_src['mjd']},
    'seeder': cy.seeding.UTFSeeder(),
    'sig' : 'tw',
    'sig_kw': dict(box=True,  t0=test_src['mjd'], dt=[0,200], flux=cy.hyp.PowerLawFlux(2.0)),
     }

tr_box = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, dt_max=200)

this_trial = tr_box.get_one_trial(n_sig, poisson=True, seed=1)
"""
srclist = cy.utils.Sources(ra=test_src["ra"], dec=test_src["dec"])

tr = cy.get_multiflare_trial_runner(ana=ana, src=srclist)

fits = []
N_trials = 1000
for i in range(0,N_trials):
    length = np.random
    t_min = test
    src_times = np.random.uniform(t_min, t_max, 1) #These are the times of our individual events. For a different flare duration, you'll have to adjust t_min and t_max
    csky_ind = 0 #The index corresponding to which source in your cy.sources object you want to inject these events on
    inj_flares = [[src_times, csky_ind]]

    this_trial = tr.get_one_trial(injeflares=inj_flares, poisson=True, TRUTH=False, seed=i)
    this_fit = tr.get_one_fit_from_trial(this_trial)
    fits.append(this_fit)
"""
"""
Configuration = {'ana': ana11,
                 'space': "ps",
                 'time': "transient",
                 'sig': 'transient'}

cy.CONF.update(Configuration)

mjd = np.array(test_src['mjd']) - np.array([10]) # Start of box time window
t100 = np.array([10]) # Width of box time window, in days
src = cy.utils.Sources(ra=test_src['ra'], dec=test_src['dec'], deg=False, mjd=mjd, t_100=t100, sigma_t=1*[0]) 

trtr = cy.get_trial_runner(src=src)

bg = trtr.get_many_fits(1000, seed=1, mp_cpus=10)

bg = cy.dists.Chi2TSD(bg)
"""
embed()

fig, ax = plt.subplots()

h = sig.get_hist(bins=15)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(sig.n_total))

x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * sig.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(sig.ndof, sig.eta))

ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()
plt.tight_layout()
plt.savefig("test_plots/sig_time_dep_test_t0_dtmax.pdf")
plt.clf()

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg = cy.bk.get_all(
        # disk location
        '{}'.format(bg_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)



print("sens, gamma = 1.5")

tr = cy.get_trial_runner(conf=conf_box, src=src, ana=ana11, flux=cy.hyp.PowerLawFlux(gamma=2), dt_max=200, mp_cpus=20)

embed()

sens = tr.find_n_sig(
    # ts, threshold
    bg.median(),
    # beta, fraction of trials which should exceed the threshold
    0.9,
    # n_inj step size for initial scan
    n_sig_step=5,
    # this many trials at a time
    batch_size=500,
    # tolerance, as estimated relative error
    tol=.05
    )

embed()

"""
#B_space
fig, ax = plt.subplots(1, 1, figsize=(5,4))
for (i, a) in enumerate(ana11):
    print(i)
    hl.plot1d (ax, a.bg_space_param.h, crosses=True, color='k', label='histogram')
    sd = np.linspace (-1, 1, 300)
    ax.plot (sd, a.bg_space_param(sindec=sd), label='spline')
    ax.set_ylim(0)
    ax.set_title(a.plot_key)
    ax.set_xlabel(r'$\sin(\delta)$')
    ax.set_ylabel(r'probability density')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig("test_plots/time_dep_test_space.pdf")
plt.clf()

#S/B_Energy
fig, axs = plt.subplots(1, 4, figsize=(15,3))
gammas = [1,2,3,4]
axs = np.ravel(axs)
for (i, gamma) in enumerate(gammas):
    a = ana11.anas[0]
    ax = axs[i]
    eprm = a.energy_pdf_ratio_model
    ss = dict(zip(eprm.gammas, eprm.ss_hl))
    things = hl.plot2d(ax, ss[gamma].eval(bins=100),
                       vmin=1e-2, vmax=1e2, log=True, cbar=True, cmap='RdBu_r')
    ax.set_title('$\gamma$ = {}'.format(gamma))
    things['colorbar'].set_label(r'$S/B$')
    ax.set_xlabel(r'$\sin(\delta)$')
    ax.set_ylabel(r'$\log_{10}(E/{GeV})$')
plt.tight_layout()
plt.savefig("test_plots/time_dep_test_energy.pdf")
plt.clf()
"""
"""
# t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)

plt.hist(bg.trials["t0"]+bg.trials["dt"]/2,histtype="step",bins=50)
plt.axvline(test_src['mjd'],color="red",label="source pos")
plt.xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
plt.legend(loc="best")
plt.savefig("test_plots/time_dep_t0_test3.pdf")
plt.clf()

#hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

plt.hist(bg.trials["dt"],histtype="step",bins=50)
plt.yscale('log')
plt.xlabel("dt / d")
plt.savefig("test_plots/time_dep_dt_test3.pdf")
plt.clf()


plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=50)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_dt_test3.pdf")
plt.clf()

plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=25)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_dt_test4.pdf")
plt.clf()

plt.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["ns"],bins=25)
plt.xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
plt.ylabel(r"$n_\mathrm{S}$")
plt.colorbar()
plt.savefig("test_plots/time_dep_d0_ns_test3.pdf")
plt.clf()

"""
