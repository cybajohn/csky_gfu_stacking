import numpy as np
import os
import csky as cy
from _paths import PATHS
from IPython import embed
from _loader import easy_source_list_loader as src_load
import healpy as hp

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import histlite as hl
import matplotlib.cm as cm

@np.vectorize
def find_n_sig(ids, sig, bg, trs, beta=0.9, nsigma=None, show_result=False):
    # get signal trials, background distribution, and trial runner
    sig_trials = cy.bk.get_best(sig,'sig')
    #sig_trials = sig
    #print(sig_trials)
    b = bg
    tr = cy.bk.get_best(trs,ids)
    # determine ts threshold
    if nsigma is not None:
        ts = b.isf_nsigma(nsigma)
    else:
        ts = b.median()
    # include background trials in calculation
    print("ts: ",ts)
    trials = {0: b.trials}
    trials.update(sig_trials)
    #print(trials)
    #trials = {"nsig":trials}
    #print(trials["0.0"])
    # get number of signal events
    # (arguments prevent additional trials from being run)
    result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=False, trials=trials, n_bootstrap=1)
    # return flux
    if show_result:
        return tr.to_E2dNdE(result, E0=100, unit=1e3)*1e3, result # at 100TeV in GeV / cm²
    return tr.to_E2dNdE(result, E0=100, unit=1e3)*1e3



def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

srcs = src_load()

ana_dir = os.path.join(PATHS.data, "ana_cache", "bg_time_dep_t0_dt_ran_new")
ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)



t_max = ana11.mjd_max
t_min = ana11.mjd_min


srcs = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran_new", "bg", "src")

src_id = np.sort(os.listdir(bg_dir))

n_srcs = len(src_id)

sources = src_load()

signals = [src["signal"] for src in srcs]
signals_all = [src["signal"] for src in sources]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)
signals_mask_2 = np.in1d(signals_all,signals_used)

src_id_all = np.reshape(np.argwhere(signals_mask_2 == True), n_srcs)

src_stds = {}
for _id_ in src_id_all:
    src_file = os.path.join(sources[_id_]["map_path"])
    skymap, header = hp.read_map(src_file,h=True, verbose=False)
    header = dict(header)
    _std = []
    _std.append(np.sin(((header["DEC_ERR_PLUS"]+header["DEC"])/180)*np.pi))
    _std.append(np.sin(((-header["DEC_ERR_MINUS"]+header["DEC"])/180)*np.pi))
    src_stds.update({_id_:_std})


#time_windows = list(np.sort(os.listdir(os.path.join(bg_dir,src_id[0]))))
conf_box = []
for ids in src_id:
    conf_box.append({
        'time': 'utf',
        'box': True,
        'fitter_args': {'t0': srcs[int(ids)]['mjd']},
        'seeder': cy.seeding.UTFSeeder(),
        'sig' : 'tw',
        'sig_kw': dict(box=True,  t0=srcs[int(ids)]['mjd'], dt=200, flux=cy.hyp.PowerLawFlux(2.0)),
         })


trs = {ids:cy.get_trial_runner(conf=conf_box[i],src=cy.sources(ra=srcs[int(ids)]["ra"], dec=srcs[int(ids)]["dec"]),flux=cy.hyp.PowerLawFlux(gamma=2),ana=ana11, dt_max=200) for i,ids in enumerate(src_id)}


sig_trials_dir = os.path.join(PATHS.data, "sig_trials_time_dep_t0_dt_gamma_ran_new")

sig_dir = cy.utils.ensure_dir('{}/sig/time_dep/gamma/2.0/src'.format(sig_trials_dir))


testbg = cy.bk.get_all(
            # disk location
            '{}/{}'.format(bg_dir,10),
            # filename pattern
            'trials*npy',
            # how to combine items within each directory
            merge=np.concatenate,
            # what to do with items after merge
            post_convert=ndarray_to_Chi2TSD)
"""
sig = cy.bk.get_all(
        # disk location
        '{}'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
    )
"""
print("testbg")
#embed()

#embed()




n_cols = 3
n_rows = int(np.ceil(n_srcs/n_cols))
fontsize = 10
n_bins = 50
max_bin = 90
_bins = np.linspace(0, max_bin, n_bins)

#from IPython import embed
#embed()

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

all_bg = []
all_sens_sig = []
all_disc_sig = []
all_sens = []
all_disc = []
all_sens_res = []
all_disc_res = []
for i,src in enumerate(src_id):
    bg = cy.bk.get_all(
        # disk location
        '{}/{}'.format(bg_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)
    all_bg.append(bg)
    sig = cy.bk.get_all(
        # disk location
        '{}/{}'.format(sig_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
    )
    sens = find_n_sig(src,sig,bg,trs,show_result=True)
    disc = find_n_sig(src,sig,bg,trs,beta=0.5, nsigma=5, show_result=True)
    all_sens_res.append(sens[1])
    all_disc_res.append(disc[1])
    all_sens.append(sens[0])
    all_disc.append(disc[0])
    print(src,sens)
    fit = sens[1].item()["info"]["n_sig"]
    fit_d = disc[1].item()["info"]["n_sig"]
    sig_list_d = disc[1].item()["info"]["n_sigs"]
    sig_list = sens[1].item()["info"]["n_sigs"]
    nearest_sig_id_d = np.searchsorted(sig_list_d,fit_d) - 1
    nearest_sig_id = np.searchsorted(sig_list,fit) - 1
    nearest_sig_d = sig_list_d[nearest_sig_id_d]
    nearest_sig = sig_list[nearest_sig_id]
    disc_sig = ndarray_to_Chi2TSD(sig["sig"][nearest_sig_d])
    sens_sig = sig["sig"][nearest_sig]
    sens_sig = ndarray_to_Chi2TSD(sens_sig)
    all_disc_sig.append(disc_sig)
    all_sens_sig.append(sens_sig)
    #embed()
    #all_values.extend(bg.values)
    #all_norms.append(bg.get_hist(bins=_bins).integrate().values)

    row_index = int(np.floor(i/n_cols))
    col_index = int(i%n_cols)
    print(row_index)
    print(col_index)
    ax = axs[row_index,col_index]
    #ax.set_box_aspect(1)
    
    sigma_5 = bg.isf_nsigma(5)
    sigma_3 = bg.isf_nsigma(3)
    h_sig = sens_sig.get_hist(bins=_bins)
    h = bg.get_hist(bins=_bins)
    hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials'.format(sens_sig.n_total))
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))
    x_sig = h_sig.centers[0]
    x = h.centers[0]
    norm_sig = h_sig.integrate().values
    norm = h.integrate().values
    ax.semilogy(x_sig, norm_sig * sens_sig.pdf(x_sig), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(sens_sig.ndof, sens_sig.eta))
    ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
    ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_ylim(10**(-1),2*bg.n_total)
    ax.set_xlabel(r'TS')
    ax.set_ylabel(r'number of trials')
    ax.legend()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sig_t0_dt_gamma_ran.pdf")
plt.clf()


print("plotting sens, disc")

decs = [np.sin(sources[src_id_]["dec"]) for src_id_ in src_id_all]

fig = plt.figure(figsize=(5,5))

_fontsize = 10
_markersize = 10
_labelsize = 10
plt.plot(decs,all_sens,'x',label=r'sens, $90\%$ at $1\mathrm{TeV}$',markersize=_markersize)
plt.plot(decs,all_disc,'x',label=r'disc, $50\%$, $5\sigma$ at $1\mathrm{TeV}$',markersize=_markersize)
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--')
plt.xlim(-1.1,1.1)
plt.xlabel(r'$\sin{(\delta)}$',fontsize=_fontsize)
plt.ylabel(r'$E^2\phi_{100\mathrm{TeV}}[\mathrm{GeV}\mathrm{cm}^{-2}]$',fontsize=_fontsize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/time_dep_sens_disc_dec.pdf")
plt.clf()

print("plotting sens, disc with time")

decs = [np.sin(sources[src_id_]["dec"]) for src_id_ in src_id_all]
times = [sources[src_id_]["mjd"] for src_id_ in src_id_all]

fig = plt.figure(figsize=(8,5))

_fontsize = 10
_markersize = 10
_labelsize = 10

cmap__ = cm.get_cmap('viridis', 9)

_xerr_min =np.array(decs)-np.array([src_stds[item][1] for item in src_stds.keys()])
_xerr_max = -np.array(decs)+np.array([src_stds[item][0] for item in src_stds.keys()])
_xerr = [_xerr_min,_xerr_max]
"""
plt.errorbar(decs, all_sens, xerr=_xerr, fmt=".", color="k")
plt.errorbar(decs, all_disc, xerr=_xerr, fmt=".", color="k")
"""
im = plt.scatter(decs,
                all_sens,
                marker='o',
                s=1,
                c=np.array(times),
                cmap=cmap__,
                vmin=t_min,
                vmax=t_max)
plt.scatter(decs,all_disc,marker='x',s=1,c=np.array(times),cmap=cmap__,vmin=t_min,vmax=t_max)
plt.scatter(decs,
                all_sens,
                marker='o',
                s=[10000*np.amax(abs(np.array(item))) for item in np.reshape(np.transpose(_xerr),(10,2))],
                c="grey",
                alpha=0.5)
plt.scatter(decs,all_disc,marker='o',s=[10000*np.amax(abs(np.array(item))) for item in np.reshape(np.transpose(_xerr),(10,2))],c="grey",alpha=0.5)
im = plt.scatter(decs,
                all_sens,
                marker='o',
                s=15,
                label=r'sens, $90\%$ at $100\mathrm{TeV}$',
                c=np.array(times),
                cmap=cmap__,
                vmin=t_min,
                vmax=t_max)
plt.scatter(decs,all_disc,marker='x',s=15,label=r'disc, $50\%$, $5\sigma$ at $100\mathrm{TeV}$',c=np.array(times),cmap=cmap__,vmin=t_min,vmax=t_max)
"""
for i,_id__ in enumerate(src_id_all):
    circle1 = plt.Circle((decs[i], all_sens[i]), np.amax(src_stds[_id__]),alpha=0.5, color='r')
    plt.gca().add_patch(circle1)
for i,_id__ in enumerate(src_id_all):
    circle1 = plt.Circle((decs[i], all_disc[i]), np.amax(src_stds[_id__]),alpha=0.5, color='r')
    plt.gca().add_patch(circle1)
"""
cbar = plt.colorbar(im)
cbar.set_label(r"Source event time in $\mathrm{mjd}$")
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--')
plt.xlim(-1.1,1.1)
plt.xlabel(r'$\sin{(\delta)}$',fontsize=_fontsize)
plt.ylabel(r'$E^2\phi_{100\mathrm{TeV}}[\mathrm{GeV}\mathrm{cm}^{-2}]$',fontsize=_fontsize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/time_dep_sens_disc_dec_time.pdf")
plt.clf()

# same plot but smaller dec window

plt.errorbar(decs, all_sens, xerr=_xerr, fmt="o", markersize=1, color="k",zorder=1)
plt.errorbar(decs, all_disc, xerr=_xerr, fmt="x", markersize=1, color="k",zorder=1)


im = plt.scatter(decs,
                all_sens,
                marker='o',
                s=25,
                label=r'sens, $90\%$ at $100\mathrm{TeV}$',
                c=np.array(times),
                cmap=cmap__,
                vmin=t_min,
                vmax=t_max,
                zorder=2)
plt.scatter(decs,all_disc,marker='x',s=25,label=r'disc, $50\%$, $5\sigma$ at $100\mathrm{TeV}$',c=np.array(times),cmap=cmap__,vmin=t_min,vmax=t_max,zorder=2)
#time_colors = [cmap__(_c) for _c in times]
#for i,item in enumerate(times):
#    plt.errorbar(decs[i], all_sens[i], xerr=[[_item_] for _item_ in np.reshape(np.transpose(_xerr),(10,2))[i]], fmt="o", color=time_colors[i])
#    plt.errorbar(decs[i], all_disc[i], xerr=[[_item_] for _item_ in np.reshape(np.transpose(_xerr),(10,2))[i]], fmt="x", color=time_colors[i])
cbar = plt.colorbar(im)
cbar.set_label(r"Source event time in $\mathrm{mjd}$")
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--',zorder=0)
plt.xlabel(r'$\sin{(\delta)}$',fontsize=_fontsize)
plt.ylabel(r'$E^2\phi_{100\mathrm{TeV}}[\mathrm{GeV}\mathrm{cm}^{-2}]$',fontsize=_fontsize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/time_dep_sens_disc_dec_time_2.pdf")
plt.clf()


print("make sens,disc time plot")

#get start and stop times of datasets, sry for the stupid code, have an appointment later
_user_path = os.path.join("/data","user","jkollek","csky_ehe_stacking","rawout_tests","cleaned_datasets_new")
datasets = [os.path.join(_user_path,'IC86_2011_exp.npy'),
        os.path.join(_user_path,'IC86_2012_exp.npy'),
        os.path.join(_user_path,'IC86_2013_exp.npy'),
        os.path.join(_user_path,'IC86_2014_exp.npy'),
        os.path.join(_user_path,'IC86_2015_exp.npy'),
        os.path.join(_user_path,'IC86_2016_exp.npy'),
        os.path.join(_user_path,'IC86_2017_exp.npy'),
        os.path.join(_user_path,'IC86_2018_exp.npy'),
        os.path.join(_user_path,'IC86_2019_exp.npy')]
start_stop = []
for data in datasets:
    _data = np.load(data)
    _start = np.amin(_data["time"])
    _stop = np.amax(_data["time"])
    _ss = [_start,_stop]
    start_stop.append(_ss)

_colors=["red","blue","red","blue","red","blue","red","blue","red"]
cmap__ = cm.get_cmap("viridis")
im = plt.errorbar(times,
                all_sens,
                xerr=[100 for item in times],
                fmt='o',
                markersize=1,
                zorder=2)
plt.errorbar(times,all_disc,xerr=[100 for item in times],fmt='x',markersize=1,zorder=2)
im = plt.scatter(times,
                all_sens,
                marker='o',
                s=25,
                label=r'sens, $90\%$ at $100\mathrm{TeV}$',
                c=np.array(decs),
                cmap=cmap__,
                vmin=np.amin(decs),
                vmax=np.amax(decs),
                zorder=3)
plt.scatter(times,all_disc,marker='x',s=25,label=r'disc, $50\%$, $5\sigma$ at $100\mathrm{TeV}$',c=np.array(decs),cmap=cmap__,vmin=np.amin(decs),vmax=np.amax(decs),zorder=3)

for i,_start_stop in enumerate(start_stop):
    plt.axvspan(_start_stop[0],_start_stop[1],alpha=0.3,color=_colors[i],zorder=0)
cbar = plt.colorbar(im)
cbar.set_label(r"Source position in $\sin{(\delta)}$")
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--',zorder=1)
plt.xlabel(r"Source event time in $\mathrm{mjd}$",fontsize=_fontsize)
plt.ylabel(r'$E^2\phi_{100\mathrm{TeV}}[\mathrm{GeV}\mathrm{cm}^{-2}]$',fontsize=_fontsize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/time_dep_sens_disc_time.pdf")

plt.clf()

im = plt.errorbar(times,
                decs,
                xerr=[100 for item in times],
                yerr=_xerr,
                fmt='o',
                markersize=1,
                zorder=2)
im = plt.scatter(times,
                decs,
                marker='o',
                s=25,
                label=r'sens, $90\%$ at $100\mathrm{TeV}$',
                c=np.array(all_sens),
                cmap=cmap__,
                vmin=np.amin(all_sens),
                vmax=np.amax(all_sens),
                zorder=3)

for i,_start_stop in enumerate(start_stop):
    plt.axvspan(_start_stop[0],_start_stop[1],alpha=0.3,color=_colors[i],zorder=0)
cbar = plt.colorbar(im)
cbar.set_label(r'sens in $E^2\phi_{100\mathrm{TeV}}[\mathrm{GeV}\mathrm{cm}^{-2}]$')
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.tick_params(axis='both', which='minor', labelsize=_labelsize)
plt.grid(ls='--',zorder=1)
plt.xlabel(r"Source event time in $\mathrm{mjd}$",fontsize=_fontsize)
plt.ylabel(r"Source position in $\sin{(\delta)}$",fontsize=_fontsize)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("test_plots/time_dep_sens_time_dec.pdf")

plt.clf()


print("make some tables")

def num_of_zeros(n):
  s = '{:.16f}'.format(n).split('.')[1]
  return len(s) - len(s.lstrip('0'))

table = ""
for i,_id in enumerate(src_id_all):
    zeros_sens = num_of_zeros(all_sens[i])
    zeros_disc = num_of_zeros(all_disc[i])
    _sens = "\\" +"num{"  + "{:.2f}".format(all_sens[i] * 10**(zeros_sens+1))+"e-{}".format(zeros_sens+1)+"}"
    _disc = "\\" +"num{"  + "{:.2f}".format(all_disc[i] * 10**(zeros_disc+1))+"e-{}".format(zeros_disc+1)+"}"
    table = table + "{}".format(int(_id+1)) + " & "+ "{:.3f}".format(decs[i]) + " & " + "{:.2f}".format(all_sens_res[i].item()["n_sig"]) + " & " + _sens + " & " + "{:.2f}".format(all_disc_res[i].item()["n_sig"]) + " & " + _disc + " \\\ "

sens_disc_file = open("tables/time_dep_sens_disc_table.tex", "w")
n = sens_disc_file.write(table)
sens_disc_file.close()



all_used_signal_params = np.sort(list(all_sens_res[0].item()["tss"].keys()))

table = "{}".format(src_id_all[0] + 1)
for i,_src_id in enumerate(src_id_all[1:]):
    table = table + " & " + "{}".format(_src_id +1)
table = table + " \\\ "
src_file = open("tables/trials_sig_time_dep_src_table.tex", "w")
n = src_file.write(table)
src_file.close()


table = ""
for s,_signal in enumerate(all_used_signal_params):
    table = table + "{:.1f}".format(_signal)
    for i,_src_id in enumerate(src_id_all):
        try:
            table = table + " & " + "{}".format(len(all_sens_res[i].item()["tss"][_signal]))
        except KeyError:
            table = table + " & " + "{}".format(0)
    table = table + " \\\ "
trials_file = open("tables/trials_sig_time_dep_table.tex", "w")
n = trials_file.write(table)
trials_file.close()



print("plotting sig sens ts")

n_cols = 3
n_rows = 4

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = axs.ravel()
for i,item in enumerate(all_sens_sig):
    ax = axs[i]
    y_ticks = [1e6,1e4,1e2,1e0]
    sigma_5 = all_bg[i].isf_nsigma(5)
    sigma_3 = all_bg[i].isf_nsigma(3)
    h_sig = item.get_hist(bins=_bins)
    h = all_bg[i].get_hist(bins=_bins)
    hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials'.format(item.n_total))
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(all_bg[i].n_total))
    x_sig = h_sig.centers[0]
    x = h.centers[0]
    norm = h.integrate().values
    ax.semilogy(x, norm * all_bg[i].pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(all_bg[i].ndof, all_bg[i].eta))
    ax.text(60,1e3,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=14)
    #ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    #ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_ylim(10**(-1),2*all_bg[i].n_total)
    ax.set_xlabel(r'TS',fontsize=14)
    ax.set_yticks(y_ticks)
    ax.tick_params(labelsize=12)
    if(i==0 or i==3 or i==6):
        ax.set_ylabel(r'number of trials',fontsize=14)
    ax.legend(fontsize=14)
axs[-1].set_visible(False)
#axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$3\sigma$",alpha=0.5) #for the label
#axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$5\sigma$",alpha=0.8)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 14},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sig_sens_ts.pdf")
plt.clf()

print("plotting sig disc ts")

n_cols = 3
n_rows = 4

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = axs.ravel()
for i,item in enumerate(all_disc_sig):
    ax = axs[i]
    sigma_5 = all_bg[i].isf_nsigma(5)
    sigma_3 = all_bg[i].isf_nsigma(3)
    h_sig = item.get_hist(bins=_bins)
    h = all_bg[i].get_hist(bins=_bins)
    hl.plot1d(ax, h_sig, crosses=True, label='{} sig trials'.format(item.n_total))
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(all_bg[i].n_total))
    x_sig = h_sig.centers[0]
    x = h.centers[0]
    norm = h.integrate().values
    ax.semilogy(x, norm * all_bg[i].pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(all_bg[i].ndof, all_bg[i].eta))
    ax.text(60,10,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=14)
    ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_ylim(10**(-1),2*all_bg[i].n_total)
    ax.set_xlabel(r'TS', fontsize=15)
    ax.set_yticks(y_ticks)
    ax.tick_params(labelsize=12)
    if(i==0 or i==3 or i==6):
        ax.set_ylabel(r'number of trials',fontsize=15)
    ax.legend(fontsize=14)
axs[-1].set_visible(False)
axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$3\sigma$",alpha=0.5) #for the label
axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$5\sigma$",alpha=0.8)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 14},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sig_disc_ts.pdf")
plt.clf()


print("plotting time windows")

n_cols = 4
n_rows = len(all_bg)
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

#all_time_mids = np.concatenate([b.trials["t0"] for b in all_bg])
#all_time_mids_min = np.amin(all_time_mids)
#all_time_mids_max = np.amax(all_time_mids)
all_time_length = np.concatenate([b.trials["dt"] for b in all_sens_sig])
all_time_length_min = np.amin(all_time_length)
all_time_length_max = np.amax(all_time_length)
all_ns = np.concatenate([b.trials["ns"] for b in all_sens_sig])
all_ns_min = np.amin(all_ns)
all_ns_max = np.amax(all_ns)

all_time_length_d = np.concatenate([b.trials["dt"] for b in all_disc_sig])
all_time_length_min_d = np.amin(all_time_length_d)
all_time_length_max_d = np.amax(all_time_length_d)
all_ns_d = np.concatenate([b.trials["ns"] for b in all_disc_sig])
all_ns_min_d = np.amin(all_ns_d)
all_ns_max_d = np.amax(all_ns_d)


n_bins = 50

#plot_3_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
#               np.linspace(all_time_length_min,all_time_length_max,n_bins+1)]

plot_4_bins = [np.linspace(all_time_length_min,all_time_length_max,n_bins+1),
               np.linspace(all_ns_min,all_ns_max,n_bins+1)]

plot_4_bins_d = [np.linspace(all_time_length_min_d,all_time_length_max_d,n_bins+1),
               np.linspace(all_ns_min_d,all_ns_max_d,n_bins+1)]

for i,bg in enumerate(all_sens_sig):
    ax = axs[i,0]
    sigma_5 = bg.isf_nsigma(5)
    sigma_3 = bg.isf_nsigma(3)
    h = bg.get_hist(bins=_bins)
    disc_sig = all_disc_sig[i]
    h_d = disc_sig.get_hist(bins=_bins)
    hl.plot1d(ax, h, crosses=True, label='{} sig trials sens'.format(bg.n_total))
    hl.plot1d(ax, h_d, crosses=True, label='{} sig trials disc'.format(disc_sig.n_total))
    x = h.centers[0]
    x_d = h_d.centers[0]
    norm = h.integrate().values
    norm_d = h_d.integrate().values
    ax.semilogy(x_d, norm_d * disc_sig.pdf(x_d), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(disc_sig.ndof, disc_sig.eta))
    ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
    #ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    #ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_xlabel(r'TS')
    ax.set_ylabel(r'number of trials')
    ax.legend()

    # t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)
    # when box_mode center (def), t0 is the middle of the box
    ##ax.hist(bg.trials["t0"],histtype="step",bins=50)
    #ax.axvline(srcs[int(src_id[i])]['mjd'],color="red",label="ra: {}, dec: {}".format(srcs[int(src_id[i])]["ra"],srcs[int(src_id[i])]["dec"]))
    ##ax.text(x=56500,y=5000,s="src pos"+"\n"+ "ra: {}, dec: {}".format(np.round(srcs[int(src_id[i])]["ra"],decimals=3),np.round(srcs[int(src_id[i])]["dec"],decimals=3)))
    ##ax.set_xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
    #ax.legend()
    #plt.savefig("test_plots/time_dep_t0_test3.pdf")
    #plt.clf()
    
    #hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)
    
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax = axs[i,1]
    dt_bins = 2.*np.logspace(-2,2,50)
    ax.hist(disc_sig.trials["dt"],histtype="step",bins=dt_bins)
    ax.hist(bg.trials["dt"],histtype="step",bins=dt_bins)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.text(x=10**(-1),y=5*10**3,s="ra: {}, dec: {}, t0: {}".format(np.round(srcs[int(src_id[i])]["ra"],decimals=3),np.round(srcs[int(src_id[i])]["dec"],decimals=3),np.round(srcs[int(src_id[i])]["mjd"],decimals=0))+"\n"+ r"sens: {}".format(np.round(all_sens[i],decimals=6)) + "$\mathrm{GeV}\mathrm{cm}^{-2}$"+"\n"+ r"disc: {}".format(np.round(all_disc[i],decimals=6))+"$\mathrm{GeV}\mathrm{cm}^{-2}$")
    ax.set_xlabel("dt / d")
    #plt.savefig("test_plots/time_dep_dt_test3.pdf")
    #plt.clf()
    
    #ax = axs[i,2]
    #im = ax.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=50)
    #ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    #ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    #plt.colorbar(im,ax)
    #plt.savefig("test_plots/time_dep_d0_dt_test3.pdf")
    #plt.clf()
    
    #ax = axs[i,2]
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.1)
    #im = ax.hist2d(bg.trials["t0"],bg.trials["dt"],bins=plot_3_bins, norm=LogNorm())
    #ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    #ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    #fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[-1],ax, orientation='vertical', pad=0.1)
    #plt.savefig("test_plots/time_dep_d0_dt_test4.pdf")
    #plt.clf()

    ax = axs[i,2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.hist2d(bg.trials["dt"],bg.trials["ns"],bins=plot_4_bins_d, norm=LogNorm())
    ax.set_xlabel(r"$dt\:/\: \mathrm{d} \: sens \: sig$")
    ax.set_ylabel(r"$n_\mathrm{S}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    
    ax = axs[i,3]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.hist2d(disc_sig.trials["dt"],disc_sig.trials["ns"],bins=plot_4_bins_d, norm=LogNorm())
    ax.set_xlabel(r"$dt\:/\: \mathrm{d} \: disc \: sig$")
    ax.set_ylabel(r"$n_\mathrm{S}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[3],ax)
    #plt.savefig("test_plots/time_dep_d0_ns_test3.pdf")
    #plt.clf()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sig_timewindows_fixed_t0_dt_gamma_ran.pdf")
plt.clf()



n_cols = 3
n_rows = 4

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = axs.ravel()

for i,_sig in enumerate(all_sens_sig):
    dt_bins = 2.*np.logspace(-2,2,50)
    ax = axs[i]
    ax.hist(_sig.trials["dt"],histtype="step",bins=dt_bins)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(7e-1,2e4)
    ax.grid('on', linestyle='--', alpha=0.5)
    ax.text(0.1,1e3,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=14)
    if(i==9 or i==8 or i==7):
        ax.set_xlabel(r'dt in $\mathrm{d}$')
    if(i==0 or i==3 or i==6 or i==9):
        ax.set_ylabel(r'number of trials')
    else:
        ax.set_yticklabels([])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
axs[-1].set_visible(False)
axs[-2].hist([10],histtype="step",bins=dt_bins,label=r"time window lengths") #for the label
axs[-2].hist([10],histtype="step",linewidth=2,bins=dt_bins,color="white")
axs[-2].set_xlim(190,200)
axs[-2].set_ylim(2,3)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sens_dt.pdf")
plt.clf()

fig = plt.figure(figsize=(5,5))
dt_bins = 2*np.logspace(-2,2,50)
plt.hist(all_sens_sig[0].trials["dt"],histtype="step",bins=dt_bins,label=r"time window lengths")
plt.yscale('log')
plt.xscale('log')
plt.grid('on',linestyle='--',alpha=.5)
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r'number of trials')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_sens_dt_1.pdf")
plt.clf()


n_cols = 3
n_rows = 4

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = axs.ravel()

for i,_sig in enumerate(all_disc_sig):
    dt_bins = 2.*np.logspace(-2,2,50)
    ax = axs[i]
    ax.hist(_sig.trials["dt"],histtype="step",bins=dt_bins)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid('on', linestyle='--', alpha=0.5)
    ax.text(0.1,1e3,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=14)
    ax.set_ylim(7e-1,2e4)
    if(i==9 or i==8 or i==7):
        ax.set_xlabel(r'dt in $\mathrm{d}$')
    if(i==0 or i==3 or i==6 or i==9):
        ax.set_ylabel(r'number of trials')
    else:
        ax.set_yticklabels([])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
axs[-1].set_visible(False)
axs[-2].hist([10],histtype="step",bins=dt_bins,label=r"time window lengths") #for the label
axs[-2].hist([10],histtype="step",linewidth=2,bins=dt_bins,color="white")
axs[-2].set_xlim(190,200)
axs[-2].set_ylim(2,3)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_disc_dt.pdf")
plt.clf()

fig = plt.figure(figsize=(5,5))
dt_bins = 2*np.logspace(-2,2,50)
plt.hist(all_disc_sig[0].trials["dt"],histtype="step",bins=dt_bins,label=r"time window lengths")
plt.yscale('log')
plt.xscale('log')
plt.grid('on',linestyle='--',alpha=.5)
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r'number of trials')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_disc_dt_1.pdf")
plt.clf()

ncols = 2
nrows = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(9.6 ,4))
axs = axs.ravel()
ax = axs[1]
ax.hist(all_disc_sig[0].trials["dt"], histtype="step", bins=dt_bins,label=r"disc time window lengths")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1,2*10**4)
plt.suptitle(r"Source Nr. " + str(src_id_all[0]+1))
ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
ax.legend(loc='upper left')
ax = axs[0]
ax.hist(all_sens_sig[0].trials["dt"],histtype="step",bins=dt_bins,label=r"sens time window lengths")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1,2*10**4)
ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
ax.set_ylabel(r"number of trials")
ax.legend(loc='upper left')
plt.savefig("test_plots/9_years_gfu_gold_disc_sens_time_dep_dt_1.pdf")
plt.clf()


print("plot time windows with ns")

ncols = 4
nrows = 3
fig, axs = plt.subplots(nrows, ncols, figsize=(13 ,8))
axs = axs.ravel()
x_ticks = [0,100,200]
for i,_sig in enumerate(all_sens_sig):
    ax = axs[i]
    im = ax.hist2d(_sig.trials["dt"],_sig.trials["ns"],bins=plot_4_bins, norm=LogNorm())
    ax.set_title("Nr. " + str(src_id_all[i]+1))
    #ax.set_box_aspect(1)
    ax.set_xticks(x_ticks)
    if i == 9 or i == 8 or i == 7 or i == 6:
        ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
    else:
        ax.set_xticklabels([])
    if i == 0 or i == 4 or i == 8:
        ax.set_ylabel(r"$\hat{n}_\mathrm{S}$")
    else:
        ax.set_yticklabels([])
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
axs[-1].set_visible(False)
plt.colorbar(im[-1],ax=axs.tolist(),label=r"number of trials")
plt.savefig("test_plots/time_window_ns_sens_time_dep.pdf")

plt.clf()

fig = plt.figure(figsize=(6,5))
im = plt.hist2d(all_sens_sig[0].trials["dt"],all_sens_sig[0].trials["ns"],bins=plot_4_bins,norm=LogNorm())
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r"$\hat{n}_\mathrm{S}$")
plt.colorbar(im[-1],label=r"number of trials")
plt.tight_layout()
plt.savefig("test_plots/time_windows_ns_sens_time_dep_1.pdf")
plt.clf()



ncols = 4
nrows = 3
fig, axs = plt.subplots(nrows, ncols, figsize=(13 ,8))
axs = axs.ravel()
x_ticks = [0,100,200]
for i,_sig in enumerate(all_disc_sig):
    ax = axs[i]
    im = ax.hist2d(_sig.trials["dt"],_sig.trials["ns"],bins=plot_4_bins_d, norm=LogNorm())
    ax.set_title("Nr. " + str(src_id_all[i]+1))
    #ax.set_box_aspect(1)
    ax.set_xticks(x_ticks)
    if i == 9 or i == 8 or i == 7 or i == 6:
        ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
    else:
        ax.set_xticklabels([])
    if i == 0 or i == 4 or i == 8:
        ax.set_ylabel(r"$\hat{n}_\mathrm{S}$")
    else:
        ax.set_yticklabels([])
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
axs[-1].set_visible(False)
plt.colorbar(im[-1],ax=axs.tolist(),label=r"number of trials")
plt.savefig("test_plots/time_window_ns_disc_time_dep.pdf")

plt.clf()

fig = plt.figure(figsize=(6,5))
im = plt.hist2d(all_disc_sig[0].trials["dt"],all_disc_sig[0].trials["ns"],bins=plot_4_bins_d,norm=LogNorm())
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r"$\hat{n}_\mathrm{S}$")
plt.colorbar(im[-1],label=r"number of trials")
plt.tight_layout()
plt.savefig("test_plots/time_windows_ns_disc_time_dep_1.pdf")
plt.clf()

ncols = 2
nrows = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(9.6 ,4))
axs = axs.ravel()
ax = axs[1]
im = ax.hist2d(all_disc_sig[0].trials["dt"],all_disc_sig[0].trials["ns"],bins=plot_4_bins_d,norm=LogNorm(vmin=1,vmax=2*10**3))
plt.suptitle(r"Source Nr. " + str(src_id_all[0]+1) + "                  ")
ax.set_title(r"Discovery Potential")
ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
ax = axs[0]
im = ax.hist2d(all_sens_sig[0].trials["dt"],all_sens_sig[0].trials["ns"],bins=plot_4_bins_d,norm=LogNorm(vmin=1,vmax=2*10**3))
ax.set_xlabel(r"$dt$ in $\mathrm{d}$")
ax.set_ylabel(r"$\hat{n}_\mathrm{S}$")
ax.set_title(r"Sensitivity")
plt.colorbar(im[-1],ax=axs.tolist(),label=r"number of trials")
plt.savefig("test_plots/time_window_ns_disc_sens_time_dep_1.pdf")
plt.clf()


print("plotting cdfs")

n_cols = 2
n_rows = 10

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

gamma = 2

for i,srcid in enumerate(src_id):
    ax = axs[i,0]
    n_sigs = all_sens_res[i].item()["info"]["n_sigs"]
    cls = np.array(all_sens_res[i].item()["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = all_sens_res[i].item()["info"]["spline"]
    ax.plot(x,spline(x),label="CDF sensitivity spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF sensitivity fit points")
    ax.text(n_sigs[bounds-1]/2.,0.65,r"$\gamma\:=\:{}$".format(gamma))
    ax.axhline(0.9,linestyle='--',alpha=0.7,label=r'$90\%$')
    ax.set_ylim(0.4,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$')
    ax.set_ylabel(r'CDF')
    ax.legend()

    ax = axs[i,1]
    n_sigs = all_disc_res[i].item()["info"]["n_sigs"]
    cls = np.array(all_disc_res[i].item()["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = all_disc_res[i].item()["info"]["spline"]
    ax.plot(x,spline(x),label="CDF discovery spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF discovery fit points")
    ax.text(n_sigs[bounds-1]/2.,0.3,r"$\gamma\:=\:{}$".format(gamma))
    ax.axhline(0.9,linestyle='--',alpha=0.7,label=r'$90\%$')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$')
    ax.set_ylabel(r'CDF')
    ax.legend()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_cdf.pdf")
plt.clf()

n_cols = 3
n_rows = 4


fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = np.ravel(axs)

for i,_src_id in enumerate(src_id_all):
    ax = axs[i]
    n_sigs = all_sens_res[i].item()["info"]["n_sigs"]
    cls = np.array(all_sens_res[i].item()["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = all_sens_res[i].item()["info"]["spline"]
    ax.plot(x,spline(x),label="CDF sensitivity spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF sensitivity fit points")
    ax.text(n_sigs[bounds-1]/2.,0.65,r"Nr. ${}$".format(_src_id+1),fontsize=15)
    ax.axhline(0.9,linestyle='--',alpha=0.7,c="blue",label=r'$90\%$')
    ax.set_ylim(0.4,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$',fontsize=15)
    ax.set_ylabel(r'CDF',fontsize=15)
axs[-1].set_visible(False)
axs[-2].plot([100,101],[0.5,0.5],"-",label=r"CDF sensitivity spline")
axs[-2].plot(100,0.5,"x",label=r"CDF sensitivity fit points") #for the label
axs[-2].plot([100,101],[0.5,0.5],c="blue",linestyle='--',alpha=0.7,label=r'$90\%$')
axs[-2].set_xlim(0,200)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_cdf_sens.pdf")
plt.clf()

n_cols = 3
n_rows = 4


fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = np.ravel(axs)

for i,_src_id in enumerate(src_id_all):
    ax = axs[i]
    n_sigs = all_disc_res[i].item()["info"]["n_sigs"]
    cls = np.array(all_disc_res[i].item()["info"]["CLs"])
    bounds = len(np.array(cls[cls<1]))
    bounds = bounds + 1
    if bounds >= len(cls):
        bounds = len(cls)
    x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
    spline = all_disc_res[i].item()["info"]["spline"]
    ax.plot(x,spline(x),label="CDF discovery spline")
    ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF discovery fit points")
    ax.text(n_sigs[bounds-1]/10.,0.8,r"Nr. ${}$".format(_src_id+1),fontsize=15)
    ax.axhline(0.5,linestyle='--',c="blue",alpha=0.7,label=r'$50\%$')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
    ax.set_xlabel(r'$n_{S}$',fontsize=15)
    ax.set_ylabel(r'CDF',fontsize=15)
axs[-1].set_visible(False)
axs[-2].plot([100,101],[0.5,0.5],"-",label=r"CDF discovery spline")
axs[-2].plot(100,0.5,"x",label=r"CDF discovery fit points") #for the label
axs[-2].plot([100,101],[0.5,0.5],c="blue",linestyle='--',alpha=0.7,label=r'$50\%$')
axs[-2].set_xlim(0,200)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_cdf_disc.pdf")
plt.clf()

n_rows = 1
n_cols = 2

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(4*n_cols,4*n_rows))
axs = np.ravel(axs)

ax = axs[1]
n_sigs = all_disc_res[0].item()["info"]["n_sigs"]
cls = np.array(all_disc_res[0].item()["info"]["CLs"])
bounds = len(np.array(cls[cls<1]))
bounds = bounds + 1
if bounds >= len(cls):
    bounds = len(cls)
x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
spline = all_disc_res[0].item()["info"]["spline"]
ax.plot(x,spline(x),label="CDF discovery spline")
ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF discovery fit points")
plt.suptitle(r"Source Nr. ${}$".format(src_id_all[0]+1))
ax.axhline(0.5,linestyle='--',c="blue",alpha=0.7,label=r'$50\%$')
ax.set_ylim(-0.1,1.1)
ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
ax.set_xlabel(r'$n_{S}$')
ax.set_ylabel(r'CDF')
ax.legend(loc='best')

ax = axs[0]
n_sigs = all_sens_res[0].item()["info"]["n_sigs"]
cls = np.array(all_sens_res[0].item()["info"]["CLs"])
bounds = len(np.array(cls[cls<1]))
bounds = bounds + 1
if bounds >= len(cls):
    bounds = len(cls)
x = np.linspace(n_sigs[0],n_sigs[bounds-1],1000)
spline = all_sens_res[0].item()["info"]["spline"]
ax.plot(x,spline(x),label="CDF discovery spline")
ax.plot(n_sigs[:bounds], cls[:bounds], 'x', label="CDF sensitivity fit points")
ax.axhline(0.9,linestyle='--',c="blue",alpha=0.7,label=r'$90\%$')
ax.set_ylim(0.4,1.1)
ax.set_xlim(-x[-1]/10.,x[-1]+x[-1]/10.)
ax.set_xlabel(r'$n_{S}$')
ax.set_ylabel(r'CDF')
ax.legend(loc='best')

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_cdf_1.pdf")
plt.clf()


print("Done")


"""
n_cols = len(time_windows)
n_rows = int(n_srcs)
fontsize = 10
n_bins = 50
max_bin = 40
_bins = np.linspace(0, max_bin, n_bins)

#from IPython import embed
#embed()

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

for i,src in enumerate(src_id):
    for t,t_window in enumerate(time_windows):
        bg = cy.bk.get_all(
            # disk location
            '{}/{}/{}'.format(bg_dir,src,t_window),
            # filename pattern
            'trials*npy',
            # how to combine items within each directory
            merge=np.concatenate,
            # what to do with items after merge
            post_convert=ndarray_to_Chi2TSD)

        row_index = int(i)
        col_index = int(t)
        print(row_index)
        print(col_index)
        ax = axs[row_index,col_index]
        #ax.set_box_aspect(1)
        
        sigma_5 = bg.isf_nsigma(5)
        sigma_3 = bg.isf_nsigma(3)
        h = bg.get_hist(bins=_bins)
        hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))
        
        x = h.centers[0]
        norm = h.integrate().values
        ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
                label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
        ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
        ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
        ax.set_xlabel(r'TS')
        ax.set_ylabel(r'number of trials')
        ax.legend()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_box.pdf")
plt.clf()
"""

"""

bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep", "bg", "src")

src_id = os.listdir(bg_dir)

n_srcs = len(src_id)

n_cols = 3
n_rows = int(np.ceil(n_srcs/n_cols))
fontsize = 10
n_bins = 50
max_bin = 40
_bins = np.linspace(0, max_bin, n_bins)

#from IPython import embed
#embed()

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

all_bg = []
all_norms = []
all_values = []


for i,src in enumerate(src_id):
    bg = cy.bk.get_all(
        # disk location
        '{}/{}'.format(bg_dir,src),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=ndarray_to_Chi2TSD)
    all_bg.append(bg)
    all_values.extend(bg.values)
    all_norms.append(bg.get_hist(bins=_bins).integrate().values)

    row_index = int(np.floor(i/n_cols))
    col_index = int(i%n_cols)
    print(row_index)
    print(col_index)
    ax = axs[row_index,col_index]
    #ax.set_box_aspect(1)
    
    sigma_5 = bg.isf_nsigma(5)
    sigma_3 = bg.isf_nsigma(3)
    h = bg.get_hist(bins=_bins)
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))
    
    x = h.centers[0]
    norm = h.integrate().values
    ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
    ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.set_xlabel(r'TS')
    ax.set_ylabel(r'number of trials')
    ax.legend()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg.pdf")
plt.clf()



#testing....
#all_bg = all_bg[:2]

n_cols = 4
n_rows = len(all_bg)
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

all_time_mids = np.concatenate([b.trials["t0"] for b in all_bg])
all_time_mids_min = np.amin(all_time_mids)
all_time_mids_max = np.amax(all_time_mids)
all_time_length = np.concatenate([b.trials["dt"] for b in all_bg])
all_time_length_min = np.amin(all_time_length)
all_time_length_max = np.amax(all_time_length)
all_ns = np.concatenate([b.trials["ns"] for b in all_bg])
all_ns_min = np.amin(all_ns)
all_ns_max = np.amax(all_ns)

n_bins = 50

plot_3_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
               np.linspace(all_time_length_min,all_time_length_max,n_bins+1)]

plot_4_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
               np.linspace(all_ns_min,all_ns_max,n_bins+1)]

for i,bg in enumerate(all_bg):
    ax = axs[i,0]
    # t0 is the time of first flare and dt is the time diff from t0 to t1 (t1 not given)
    # when box_mode center (def), t0 is the middle of the box
    ax.hist(bg.trials["t0"],histtype="step",bins=50)
    #ax.axvline(srcs[int(src_id[i])]['mjd'],color="red",label="ra: {}, dec: {}".format(srcs[int(src_id[i])]["ra"],srcs[int(src_id[i])]["dec"]))
    ax.text(x=56500,y=5000,s="src pos"+"\n"+ "ra: {}, dec: {}".format(np.round(srcs[int(src_id[i])]["ra"],decimals=3),np.round(srcs[int(src_id[i])]["dec"],decimals=3)))
    ax.set_xlabel(r"$t_{box} \:/\: \mathrm{mjd}$")
    #ax.legend()
    #plt.savefig("test_plots/time_dep_t0_test3.pdf")
    #plt.clf()
    
    #hist, bins, _ = plt.hist(bg.trials["dt"],bins=50)
    
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax = axs[i,1]
    ax.hist(bg.trials["dt"],histtype="step",bins=50)
    ax.set_yscale('log')
    ax.set_xlabel("dt / d")
    #plt.savefig("test_plots/time_dep_dt_test3.pdf")
    #plt.clf()
    
    #ax = axs[i,2]
    #im = ax.hist2d(bg.trials["t0"]+bg.trials["dt"]/2,bg.trials["dt"],bins=50)
    #ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    #ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    #plt.colorbar(im,ax)
    #plt.savefig("test_plots/time_dep_d0_dt_test3.pdf")
    #plt.clf()
    
    ax = axs[i,2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.hist2d(bg.trials["t0"],bg.trials["dt"],bins=plot_3_bins, norm=LogNorm())
    ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    ax.set_ylabel(r"$t_{box,length}\:/\:\mathrm{d}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[-1],ax, orientation='vertical', pad=0.1)
    #plt.savefig("test_plots/time_dep_d0_dt_test4.pdf")
    #plt.clf()
    
    ax = axs[i,3]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    ax.hist2d(bg.trials["t0"],bg.trials["ns"],bins=plot_4_bins, norm=LogNorm())
    ax.set_xlabel(r"$t_{box,mid}\:/\: \mathrm{mjd}$")
    ax.set_ylabel(r"$n_\mathrm{S}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[3],ax)
    #plt.savefig("test_plots/time_dep_d0_ns_test3.pdf")
    #plt.clf()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_timewindows.pdf")
plt.clf()

"""
"""

fig, ax = plt.subplots()
values = np.sum([bgr.get_hist(bins=_bins).get_values() for bgr in all_bg],axis=0)/20
embed()
h = hl.hist_direct(all_values,bins=_bins)/20
embed()
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))

x = h.centers[0]
norm = h.integrate().values
bg_sum = np.sum(all_norm * [bgr.pdf(x) for bgr in all_bg],axis=0)/20
embed()
ax.semilogy(x, bg_sum, lw=1, ls='--',
        label=r'mean')
ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
ax.set_xlabel(r'TS')
ax.set_ylabel(r'number of trials')
ax.legend()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_mean.pdf")
plt.clf()

"""

"""
print(bg)
from IPython import embed
embed()

sigma_5 = bg.isf_nsigma(5)
sigma_3 = bg.isf_nsigma(3)

fontsize = 15

#fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(13,5))

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
plt.savefig("test_plots/9_years_gfu_gold_bg.pdf")
plt.clf()
"""
