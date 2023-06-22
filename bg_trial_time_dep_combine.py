import numpy as np
import os
import csky as cy
from _paths import PATHS
from IPython import embed
from _loader import easy_source_list_loader as src_load

import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import histlite as hl

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




n_cols = 3
n_rows = int(np.ceil(n_srcs/n_cols))
fontsize = 12
n_bins = 50
max_bin = 40
_bins = np.linspace(0, max_bin, n_bins)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

axs = axs.ravel()

all_bg = []

all_median = []
all_sigma_5 = []
all_sigma_3 = []

mpl.rcParams.update({'font.size': 12})

print("collect bg and plot ts")

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
    ax = axs[i]
    median = bg.median()
    sigma_5 = bg.isf_nsigma(5)
    sigma_3 = bg.isf_nsigma(3)
    all_median.append(median)
    all_sigma_5.append(sigma_5)
    all_sigma_3.append(sigma_3)
    
    h = bg.get_hist(bins=_bins)
    hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(bg.n_total))
    
    x = h.centers[0]
    norm = h.integrate().values
    ax.semilogy(x, norm * bg.pdf(x), lw=1, ls='--',
            label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(bg.ndof, bg.eta))
    ax.axvline(x=sigma_5,ls='--',color='grey',alpha=0.8)#,label=r'$5\sigma$')
    ax.axvline(x=sigma_3,ls='--',color='grey',alpha=0.5)#,label=r'$3\sigma$')
    ax.grid('on', linestyle='--', alpha=0.4)
    ax.text(5,0.1,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=13)
    if(i==9 or i==8 or i==7):
        ax.set_xlabel(r'TS')
    if(i==0 or i==3 or i==6 or i==9):
        ax.set_ylabel(r'number of trials')
    else:
        ax.set_yticklabels([])
    ax.legend()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
axs[-1].set_visible(False)
axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$3\sigma$",alpha=0.5) #for the label
axs[-2].plot(1e4,20,"--",markersize = .1,c="grey",label=r"$5\sigma$",alpha=0.8)
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 14},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_t0.pdf")
plt.clf()




fig, ax = plt.subplots()
h = all_bg[0].get_hist(bins=_bins)
hl.plot1d(ax, h, crosses=True, label='{} bg trials'.format(all_bg[0].n_total))
x = h.centers[0]
norm = h.integrate().values
ax.semilogy(x, norm * all_bg[0].pdf(x), lw=1, ls='--',
        label=r'$\chi^2[{:.2f} \mathrm{{dof}},\: \eta={:.3f}]$'.format(all_bg[0].ndof, all_bg[0].eta))
ax.axvline(x=all_sigma_5[0],ls='--',color='grey',alpha=0.8,label=r'$5\sigma$')
ax.axvline(x=all_sigma_3[0],ls='--',color='grey',alpha=0.5,label=r'$3\sigma$')
ax.grid('on', linestyle='--', alpha=0.4)
ax.set_ylabel(r'number of trials')
ax.set_xlabel(r'TS')
ax.set_title(r'Source Nr. ${}$'.format(src_id_all[0] + 1))
ax.legend(loc='best')
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_t0_1.pdf")
plt.clf()


print("make some tables")

def num_of_zeros(n):
  s = '{:.16f}'.format(n).split('.')[1]
  return len(s) - len(s.lstrip('0'))

table = ""
for i,_id in enumerate(src_id_all):
    #some medians are just very very close to 0
    #median = "\\" +"num{"  + "{:.3f}".format(all_median[i] * 10**(num_of_zeros(all_median[i])+1))+"e-{}".format(num_of_zeros(all_median[i])+1)+"}"
    table = table + "{}".format(int(_id+1)) + " & " + "{}".format(all_bg[i].n_total) + " & " + "{:.2f}".format(all_bg[i].ndof) + " & " + "{:.2f}".format(all_bg[i].eta) + " & " + "{:.3f}".format(all_median[i]) + " & " + "{:.3f}".format(all_sigma_3[i]) + " & " +"{:.3f}".format(all_sigma_5[i]) + " \\\ "

sigma_file = open("tables/time_dep_sigma_table.tex", "w")
n = sigma_file.write(table)
sigma_file.close()

print("plot time windows")

n_cols = 3
n_rows = 4

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))
axs = axs.ravel()

for i,bg in enumerate(all_bg):
    dt_bins = 2.*np.logspace(-2,2,50)
    ax = axs[i]
    ax.hist(bg.trials["dt"],histtype="step",bins=dt_bins)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid('on', linestyle='--', alpha=0.5)
    ax.text(0.1,1e5,r'Nr. ${}$'.format(src_id_all[i] + 1),fontsize=12)
    ax.set_ylim(1e2,1e6)
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
axs[-2].legend(loc="center",edgecolor="white",prop={'size': 13},framealpha=1)
axs[-2].set_frame_on(False)
axs[-2].set_xticks([])
axs[-2].set_yticks([])
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_dt.pdf")
plt.clf()

fig = plt.figure(figsize=(6,5))
dt_bins = 2*np.logspace(-2,2,50)
plt.hist(all_bg[0].trials["dt"],histtype="step",bins=dt_bins,label=r"bg time window lengths")
plt.yscale('log')
plt.xscale('log')
plt.grid('on',linestyle='--',alpha=.5)
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r'number of trials')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_dt_1.pdf")
plt.clf()


n_cols = 3
n_rows = len(all_bg)
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=False, figsize=(5*n_cols,4*n_rows))

#all_time_mids = np.concatenate([b.trials["t0"] for b in all_bg])
#all_time_mids_min = np.amin(all_time_mids)
#all_time_mids_max = np.amax(all_time_mids)
all_time_length = np.concatenate([b.trials["dt"] for b in all_bg])
all_time_length_min = np.amin(all_time_length)
all_time_length_max = np.amax(all_time_length)
all_ns = np.concatenate([b.trials["ns"] for b in all_bg])
all_ns_min = np.amin(all_ns)
all_ns_max = np.amax(all_ns)

n_bins = 50

#plot_3_bins = [np.linspace(all_time_mids_min,all_time_mids_max,n_bins+1),
#               np.linspace(all_time_length_min,all_time_length_max,n_bins+1)]

plot_4_bins = [np.linspace(all_time_length_min,all_time_length_max,n_bins+1),
               np.linspace(all_ns_min,all_ns_max,n_bins+1)]

all_median = []
all_sigma_5 = []
all_sigma_3 = []


for i,bg in enumerate(all_bg):
    ax = axs[i,0]
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
    ax.hist(bg.trials["dt"],histtype="step",bins=dt_bins)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.text(x=10**(-1),y=10**5,s="src pos"+"\n"+ "ra: {}, dec: {}, t0: {}".format(np.round(srcs[int(src_id[i])]["ra"],decimals=3),np.round(srcs[int(src_id[i])]["dec"],decimals=3),np.round(srcs[int(src_id[i])]["mjd"],decimals=0)))
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
    im = ax.hist2d(bg.trials["dt"],bg.trials["ns"],bins=plot_4_bins, norm=LogNorm())
    ax.set_xlabel(r"$dt\:/\: \mathrm{d}$")
    ax.set_ylabel(r"$n_\mathrm{S}$")
    fig.colorbar(im[-1], cax=cax, orientation='vertical')
    #plt.colorbar(im[3],ax)
    #plt.savefig("test_plots/time_dep_d0_ns_test3.pdf")
    #plt.clf()

plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_timewindows_fixed_t0.pdf")
plt.clf()


print("plot time windows with ns")

ncols = 4
nrows = 3
fig, axs = plt.subplots(nrows, ncols, figsize=(13 ,8))
axs = axs.ravel()
x_ticks = [0,100,200]
for i,bg in enumerate(all_bg):
    ax = axs[i]
    im = ax.hist2d(bg.trials["dt"],bg.trials["ns"],bins=plot_4_bins, norm=LogNorm())
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
plt.savefig("test_plots/time_window_ns_bg_time_dep.pdf")

plt.clf()

fig = plt.figure(figsize=(6,5))
im = plt.hist2d(all_bg[0].trials["dt"],all_bg[0].trials["ns"],bins=plot_4_bins,norm=LogNorm())
plt.title(r"Source Nr. ${}$".format(src_id_all[0]+1))
plt.xlabel(r'dt in $\mathrm{d}$')
plt.ylabel(r"$\hat{n}_\mathrm{S}$")
plt.colorbar(im[-1],label=r"number of trials")
plt.tight_layout()
plt.savefig("test_plots/time_window_ns_bg_time_dep_1.pdf")
plt.clf()


fig, axs = plt.subplots(1, 2, figsize=(13 ,5))
axs = axs.ravel()
ax = axs[0]
dt_bins = 2*np.logspace(-2,2,50)
ax.hist(all_bg[0].trials["dt"],histtype="step",bins=dt_bins,label=r"bg time window lengths")
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid('on',linestyle='--',alpha=.5)
plt.suptitle(r"Source Nr. ${}$".format(src_id_all[0]+1),fontsize=15)
ax.set_xlabel(r'dt in $\mathrm{d}$',fontsize=15)
ax.set_ylabel(r'number of trials',fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)
ax.legend(loc='upper left')

ax = axs[1]
im = ax.hist2d(all_bg[0].trials["dt"],all_bg[0].trials["ns"],bins=plot_4_bins,norm=LogNorm())
ax.set_xlabel(r'dt in $\mathrm{d}$',fontsize=15)
ax.set_ylabel(r"$\hat{n}_\mathrm{S}$",fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)
cb = plt.colorbar(im[-1])
cb.set_label(label='number of trials', size='large')
cb.ax.tick_params(labelsize='large')
plt.tight_layout()
plt.savefig("test_plots/9_years_gfu_gold_time_dep_bg_dt_and_dt_ns.pdf")
plt.clf()


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
