import numpy as np
import os
import csky as cy
from _paths import PATHS
from _loader import source_list_loader

from _loader import easy_source_list_loader as src_load


import matplotlib.pyplot as plt
import histlite as hl

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

bg_dir = os.path.join(PATHS.data, "bg_trials_new", "bg_new")
sig_dir = os.path.join(PATHS.data, "sig_trials_new_2", "sig_new")

print("load sig")

sig = cy.bk.get_all(
        # disk location
        '{}/for_gamma_3/gamma'.format(sig_dir),
        # filename pattern
        'trials*npy',
        # how to combine items within each directory
        merge=np.concatenate,
        # what to do with items after merge
        post_convert=cy.utils.Arrays
        )

from IPython import embed
#embed()


gammas = {}
for gamma in sig.keys():
	#print(gamma)
	new_entry = [[],[]]
	for n_sig in sig[gamma]['sig'].keys():
		#print(n_sig)
		new_entry[0].extend(sig[gamma]['sig'][n_sig]["ns"])
		new_entry[1].extend(sig[gamma]['sig'][n_sig]["gamma"])
	gammas.update({gamma: new_entry})

bins = np.linspace(0,30,31)
gamma_values = [[] for i in range(len(bins)-1)]
for j, ns_value in enumerate(gammas[2.0][0]):
	for i in range(len(bins)-1):
		if(ns_value >= bins[i] and ns_value < bins[i+1]):
			gamma_values[i].append(gammas[2.0][1][j])
			break

gamma_means = [np.mean(a) for a in gamma_values]
print(gamma_means)
bin_centres =(bins[1:] + bins[:-1])/2.
print(bin_centres)

direct_n_sig = {}

for gamma in sig.keys():
	gamma_mean = []
	gamma_std = []
	for n_sig in sig[gamma]['sig'].keys():
		_gamma_mean = np.mean(sig[gamma]['sig'][n_sig]['gamma'])
		_gamma_std = np.std(sig[gamma]['sig'][n_sig]['gamma'])
		gamma_mean.append(_gamma_mean)
		gamma_std.append(_gamma_std)
	direct_n_sig.update({gamma: [gamma_mean,gamma_std]})

direct_n_sig_comp = {}

for gamma in sig.keys():
        n_sig_mean = []
        n_sig_std = []
        for n_sig in sig[gamma]['sig'].keys():
                _n_sig_mean = np.mean(sig[gamma]['sig'][n_sig]['ns'])
                _n_sig_std = np.std(sig[gamma]['sig'][n_sig]['ns'])
                n_sig_mean.append(_n_sig_mean)
                n_sig_std.append(_n_sig_std)
        direct_n_sig_comp.update({gamma: [n_sig_mean,n_sig_std]})



#embed()



fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

ax = axs[0,0]
x = list(sig[2.0]['sig'].keys())
gamma = 1.5
y = direct_n_sig[gamma][0]
yerr = direct_n_sig[gamma][1]
ax.axhline(gamma,ls='--',color='cyan')
ax.errorbar(x, y, yerr=yerr, fmt='.')
ax.set_title(r'$\gamma={}$'.format(gamma))
ax.set_ylabel(r'$\hat\gamma$',rotation='horizontal')


ax = axs[0,1]
gamma = 2.0
y = direct_n_sig[gamma][0]
yerr = direct_n_sig[gamma][1]
ax.axhline(gamma,ls='--',color='cyan')
ax.errorbar(x, y, yerr=yerr, fmt='.')
ax.set_title(r'$\gamma={}$'.format(gamma))

ax = axs[1,0]
gamma = 2.5
y = direct_n_sig[gamma][0]
yerr = direct_n_sig[gamma][1]
ax.axhline(gamma,ls='--',color='cyan')
ax.errorbar(x, y, yerr=yerr, fmt='.')
ax.set_title(r'$\gamma={}$'.format(gamma))
ax.set_xlabel(r'$n_{inj}$')
ax.set_ylabel(r'$\hat\gamma$',rotation='horizontal')

ax = axs[1,1]
gamma = 3.0
y = direct_n_sig[gamma][0]
yerr = direct_n_sig[gamma][1]
ax.axhline(gamma,ls='--',color='cyan')
ax.errorbar(x, y, yerr=yerr, fmt='.')
ax.set_title(r'$\gamma={}$'.format(gamma))
ax.set_xlabel(r'$n_{inj}$')

fig.suptitle(r'set $\gamma$ vs fitted $\hat\gamma$ ')

plt.savefig('test_plots/gamma_fit.pdf')

plt.clf()

gamma_list= list(sig.keys())[1:-2]
gamma_len = len(gamma_list)
_nrows = int(np.ceil(gamma_len/2))
print("making plot with {} rows".format(_nrows))
fig, axs = plt.subplots(nrows=_nrows, ncols=2, sharex=False, sharey=True, figsize=(2*3,3*_nrows))
for i,gamma in enumerate(gamma_list):
	print('plotting gamma={} ,i={}'.format(gamma,i))
	x = list(sig[gamma]['sig'].keys()) 
	y = direct_n_sig[gamma][0]
	yerr = direct_n_sig[gamma][1]
	ax = axs[int(np.floor(i/2)),int(i%2)]
	ax.axhline(gamma,ls='--',color='cyan')
	ax.errorbar(x, y, yerr=yerr, fmt='.')
	ax.set_title(r'$\gamma={}$'.format(gamma))
	if(int(np.floor(i/2)) >= (_nrows - 1)):
		ax.set_xlabel(r'$n_{inj}$')
		plt.setp(ax.get_xticklabels(), visible=True)
	elif(gamma_len % 2 == 1 and np.floor(i/2) == _nrows -2 and i%2 == 1):
		axs[-1,-1].axis('off')
		axs[-2,-1].set_xlabel(r'$n_{inj}$')
		plt.setp(axs[-2,-1].get_xticklabels(), visible=True)
	else:
		plt.setp(ax.get_xticklabels(), visible=False)
	if(int(i%2) == 0):
		ax.set_ylabel(r'$\hat\gamma$',rotation='horizontal')
fig.suptitle(r'set $\gamma$ vs fitted $\hat\gamma$ ')
plt.savefig('test_plots/gamma_fit_auto.pdf')
plt.clf()


gamma_list= list(sig.keys())[1:-2]
gamma_len = len(gamma_list)
_nrows = int(np.ceil(gamma_len/2))
print("making plot with {} rows".format(_nrows))
fig, axs = plt.subplots(nrows=_nrows, ncols=2, sharex=False, sharey=True, figsize=(2*3,3*_nrows))
for i,gamma in enumerate(gamma_list):
	print('plotting gamma={} ,i={}'.format(gamma,i))
	x = list(sig[gamma]['sig'].keys())
	y = direct_n_sig_comp[gamma][0]
	yerr = direct_n_sig_comp[gamma][1]
	ax = axs[int(np.floor(i/2)),int(i%2)]
	ax.plot([np.amin(x),np.amax(x)], [np.amin(x),np.amax(x)], ls="--", color='cyan')
	ax.errorbar(x, y, yerr=yerr, fmt='.')
	ax.set_title(r'$\gamma={}$'.format(gamma))
	if(int(np.floor(i/2)) >= (_nrows - 1)):
		ax.set_xlabel(r'$n_{inj}$')
		plt.setp(ax.get_xticklabels(), visible=True)
	elif(gamma_len % 2 == 1 and np.floor(i/2) == _nrows -2 and i%2 == 1):
		axs[-1,-1].axis('off')
		axs[-2,-1].set_xlabel(r'$n_{inj}$')
		plt.setp(axs[-2,-1].get_xticklabels(), visible=True)
	else:
		plt.setp(ax.get_xticklabels(), visible=False)
	if(int(i%2) == 0):
		ax.set_ylabel(r'$\hat{n}_{inj}$',rotation='horizontal')
fig.suptitle(r'set $n_{inj}$ vs fitted $\hat{n}_{inj}$ ')
fig.tight_layout()
plt.savefig('test_plots/ns_fit_auto.pdf')
plt.clf()



def plot_ns_auto(sig,gamma_list,name,size=3,overbound=0.1, fontsize=10):
	gamma_len = len(gamma_list)
	all_x = []
	all_x.extend(list(sig[g]['sig'].keys()) for g in gamma_list)
	xmin = np.amin(all_x)
	xmax = np.amax(all_x)
	label_x = xmin
	xlength = xmax-xmin
	xmin-=overbound*xlength
	xmax+=overbound*xlength
	all_y_min = []
	all_y_max = []
	all_y_min.extend((np.array(direct_n_sig_comp[g][0]) - np.array(direct_n_sig_comp[g][1])) for g in gamma_list)
	all_y_max.extend((np.array(direct_n_sig_comp[g][0]) + np.array(direct_n_sig_comp[g][1])) for g in gamma_list)
	ymin = np.amin(all_y_min)
	ymax = np.amax(all_y_max)
	ylength = ymax-ymin
	label_y = ymax - overbound*ylength
	ymin -= overbound*ylength
	ymax += overbound*ylength
	_nrows = int(np.ceil(gamma_len/2))
	print("making plot with {} rows".format(_nrows))
	fig, axs = plt.subplots(nrows=_nrows, ncols=2, sharex=False, sharey=False, figsize=(2*size,size*_nrows), gridspec_kw = {'wspace':0, 'hspace':0})
	for i,gamma in enumerate(gamma_list):
		print('plotting gamma={} ,i={}'.format(gamma,i))
		x = list(sig[gamma]['sig'].keys())
		y = direct_n_sig_comp[gamma][0]
		yerr = direct_n_sig_comp[gamma][1]
		ax = axs[int(np.floor(i/2)),int(i%2)]
		ax.plot([np.amin(x),np.amax(x)], [np.amin(x),np.amax(x)], ls="--", color='black')
		ax.errorbar(x, y, yerr=yerr, fmt='.')
		ax.grid('on', linestyle='--', alpha=0.5)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(label_x,label_y,r'$\gamma={}$'.format(gamma),fontsize=fontsize,bbox={'facecolor': 'white', 'alpha': 1, 'boxstyle': 'round'})
		if(int(np.floor(i/2)) >= (_nrows - 1)):
			ax.set_xlabel(r'$n_{inj}$')
			plt.setp(ax.get_xticklabels(), visible=True)
		elif(gamma_len % 2 == 1 and np.floor(i/2) == _nrows -2 and i%2 == 1):
			axs[-1,-1].axis('off')
			axs[-2,-1].set_xlabel(r'$n_{\mathrm{inj}}$')
			plt.setp(axs[-2,-1].get_xticklabels(), visible=True)
		else:
			plt.setp(ax.get_xticklabels(), visible=False)
		if(int(i%2) == 0):
			ax.set_ylabel(r'$\hat{n}_{\mathrm{S}}$')
		else:
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='y', colors=(0,0,0,0)) #make ticks transparent
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(fontsize)
	fig.suptitle(r'set $n_{\mathrm{inj}}$ vs fitted $\hat{n}_{\mathrm{S}}$', fontsize=fontsize)
	fig.tight_layout()
	plt.savefig(name)
	plt.clf()
	return

def plot_ns_auto_for_thesis(sig,gamma_list,name,overbound=0.1, fontsize=12):
    gamma_len = len(gamma_list)
    all_x = []
    all_x.extend(list(sig[g]['sig'].keys()) for g in gamma_list)
    xmin = np.amin(all_x)
    xmax = np.amax(all_x)
    label_x = xmin
    xlength = xmax-xmin
    xmin-=overbound*xlength
    xmax+=overbound*xlength
    all_y_min = []
    all_y_max = []
    all_y_min.extend((np.array(direct_n_sig_comp[g][0]) - np.array(direct_n_sig_comp[g][1])) for g in gamma_list)
    all_y_max.extend((np.array(direct_n_sig_comp[g][0]) + np.array(direct_n_sig_comp[g][1])) for g in gamma_list)
    ymin = np.amin(all_y_min)
    ymax = np.amax(all_y_max)
    ylength = ymax-ymin
    label_y = ymax - overbound*ylength
    ymin -= overbound*ylength
    ymax += overbound*ylength
    _nrows = int(np.ceil(gamma_len/2))
    print("making plot with {} rows".format(_nrows))
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(11,10))
    axs = axs.ravel()
    for i,gamma in enumerate(gamma_list):
        print('plotting gamma={} ,i={}'.format(gamma,i))
        x = list(sig[gamma]['sig'].keys())
        y = direct_n_sig_comp[gamma][0]
        yerr = direct_n_sig_comp[gamma][1]
        ax = axs[i]
        ax.plot([np.amin(x),np.amax(x)], [np.amin(x),np.amax(x)], ls="--", color='black')
        ax.errorbar(x, y, yerr=yerr, fmt='.')
        ax.grid('on', linestyle='--', alpha=0.5)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.text(label_x,label_y,r'$\gamma={}$'.format(gamma),fontsize=fontsize)
        if(i==4 or i== 5 or i==6):
            ax.set_xlabel(r'$n_{inj}$')
            plt.setp(ax.get_xticklabels(), visible=True)
        else:
            ax.set_xticklabels([])
        if(i==0 or i==3 or i==6):
            ax.set_ylabel(r'$\hat{n}_{\mathrm{S}}$')
        else:
            ax.set_yticklabels([])
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    axs[-1].set_visible(False)
    axs[-2].plot(300,300,"--",markersize = .1,c="black",label=r"$n_{\mathrm{inj}}=\hat{n}_{\mathrm{S}}$") #for the label
    axs[-2].errorbar(300,300,yerr=[0],fmt=".",label=r"$\hat{n}_{\mathrm{S}}$")
    axs[-2].plot(300,300,"o",c="white")
    axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
    axs[-2].set_frame_on(False)
    axs[-2].set_xticks([])
    axs[-2].set_yticks([])
    fig.suptitle(r'set $n_{\mathrm{inj}}$ vs fitted $\hat{n}_{\mathrm{S}}$',fontsize=fontsize+2)
    fig.tight_layout()
    plt.savefig(name)
    plt.clf()
    return

def plot_gamma_auto_for_thesis(sig,gamma_list,name,overbound=0.1, fontsize=12):
        gamma_len = len(gamma_list)
        all_x = []
        all_x.extend(list(sig[g]['sig'].keys()) for g in gamma_list)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        label_x = xmin
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        all_y_min = []
        all_y_max = []
        all_y_min.extend((np.array(direct_n_sig[g][0]) - np.array(direct_n_sig[g][1])) for g in gamma_list)
        all_y_max.extend((np.array(direct_n_sig[g][0]) + np.array(direct_n_sig[g][1])) for g in gamma_list)
        ymin = np.amin(all_y_min)
        ymax = np.amax(all_y_max)
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(11,10))
        axs = axs.ravel()
        for i,gamma in enumerate(gamma_list):
                print('plotting gamma={} ,i={}'.format(gamma,i))
                x = list(sig[gamma]['sig'].keys())
                y = direct_n_sig[gamma][0]
                yerr = direct_n_sig[gamma][1]
                ax = axs[i]
                ax.axhline(gamma, ls="--", color='black',label=r'$\gamma={}$'.format(gamma))
                ax.errorbar(x, y, yerr=yerr, fmt='.',label=r'$\hat\gamma$')
                ax.grid('on', linestyle='--', alpha=0.5)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                ax.legend(loc='best')
                #ax.text(label_x,label_y,r'$\gamma={}$'.format(gamma),fontsize=fontsize,bbox={'facecolor': 'white', 'alpha': 1, 'boxstyle': 'round'})
                if(i==4 or i==5 or i==6):
                        ax.set_xlabel(r'$n_{inj}$')
                        plt.setp(ax.get_xticklabels(), visible=True)
                else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                if(i==0 or i==3 or i==6):
                        ax.set_ylabel(r'$\hat\gamma$')
                else:
                        plt.setp(ax.get_yticklabels(), visible=False)
                for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        axs[-1].set_visible(False)
        axs[-2].set_visible(False)
        fig.suptitle(r'set $\gamma$ vs fitted $\hat\gamma$', fontsize=fontsize+2)
        fig.tight_layout()
        plt.savefig(name)
        plt.clf()
        return


def plot_gamma_auto(sig,gamma_list,name,size=3,overbound=0.1, fontsize=10):
        gamma_len = len(gamma_list)
        all_x = []
        all_x.extend(list(sig[g]['sig'].keys()) for g in gamma_list)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        label_x = xmin
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        all_y_min = []
        all_y_max = []
        all_y_min.extend((np.array(direct_n_sig[g][0]) - np.array(direct_n_sig[g][1])) for g in gamma_list)
        all_y_max.extend((np.array(direct_n_sig[g][0]) + np.array(direct_n_sig[g][1])) for g in gamma_list)
        ymin = np.amin(all_y_min)
        ymax = np.amax(all_y_max)
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        _nrows = int(np.ceil(gamma_len/2))
        print("making plot with {} rows".format(_nrows))
        fig, axs = plt.subplots(nrows=_nrows, ncols=2, sharex=False, sharey=False, figsize=(2*size,size*_nrows), gridspec_kw = {'wspace':0, 'hspace':0})
        for i,gamma in enumerate(gamma_list):
                print('plotting gamma={} ,i={}'.format(gamma,i))
                x = list(sig[gamma]['sig'].keys())
                y = direct_n_sig[gamma][0]
                yerr = direct_n_sig[gamma][1]
                ax = axs[int(np.floor(i/2)),int(i%2)]
                ax.axhline(gamma, ls="--", color='black',label=r'$\gamma={}$'.format(gamma))
                ax.errorbar(x, y, yerr=yerr, fmt='.')
                ax.grid('on', linestyle='--', alpha=0.5)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                ax.legend(loc='best')
                #ax.text(label_x,label_y,r'$\gamma={}$'.format(gamma),fontsize=fontsize,bbox={'facecolor': 'white', 'alpha': 1, 'boxstyle': 'round'})
                if(int(np.floor(i/2)) >= (_nrows - 1)):
                        ax.set_xlabel(r'$n_{inj}$')
                        plt.setp(ax.get_xticklabels(), visible=True)
                elif(gamma_len % 2 == 1 and np.floor(i/2) == _nrows -2 and i%2 == 1):
                        axs[-1,-1].axis('off')
                        axs[-2,-1].set_xlabel(r'$n_{\mathrm{inj}}$')
                        plt.setp(axs[-2,-1].get_xticklabels(), visible=True)
                else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                if(int(i%2) == 0):
                        ax.set_ylabel(r'$\hat\gamma$')
                else:
                        plt.setp(ax.get_yticklabels(), visible=False)
                        ax.tick_params(axis='y', colors=(0,0,0,0)) #make ticks transparent
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        fig.suptitle(r'set $\gamma$ vs fitted $\hat\gamma$', fontsize=fontsize)
        fig.tight_layout()
        plt.savefig(name)
        plt.clf()
        return

plot_gamma_auto_for_thesis(sig=sig,gamma_list=list(sig.keys()),name="test_plots/gamma_fit_auto_3.pdf")
plot_ns_auto_for_thesis(sig=sig,gamma_list=list(sig.keys()),name="test_plots/ns_fit_auto_4.pdf")
plot_ns_auto(sig=sig,gamma_list=list(sig.keys())[1:-2],name='test_plots/ns_fit_auto_2.pdf',fontsize=15)
plot_ns_auto(sig=sig,gamma_list=list(sig.keys()),name='test_plots/ns_fit_auto_3.pdf',fontsize=15)
plot_gamma_auto(sig=sig,gamma_list=list(sig.keys())[1:-2],name='test_plots/gamma_fit_auto_2.pdf',fontsize=15)


# time dep 

print("time dep bias...")

sig_trials_dir = os.path.join(PATHS.data, "sig_trials_time_dep_t0_dt_gamma_ran_new")

sig_dir = cy.utils.ensure_dir('{}/sig/time_dep/gamma/2.0/src'.format(sig_trials_dir))

bg_dir = os.path.join(PATHS.data, "bg_trials_time_dep_t0_dt_ran_new", "bg", "src")

src_id = np.sort(os.listdir(bg_dir))

# get gamma and ns means with deviations

all_time_dep_gamma = []
all_time_dep_ns = []
all_ns_used = []

print("collecting...")

for i,src in enumerate(src_id):
    all_gamma = [[],[]]
    all_ns = [[],[]]
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
    for _ns in sig["sig"].keys():
        all_gamma[0].append(np.mean(sig["sig"][_ns]["gamma"]))
        all_gamma[1].append(np.std(sig["sig"][_ns]["gamma"]))
        all_ns[0].append(np.mean(sig["sig"][_ns]["ns"]))
        all_ns[1].append(np.std(sig["sig"][_ns]["ns"]))
    all_time_dep_gamma.append(all_gamma)
    all_time_dep_ns.append(all_ns)
    all_ns_used.append(list(sig["sig"].keys()))

print("fix source index...")

# just get any ana for the start and end times....
ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_new")

ana11 = cy.get_analysis(cy.selections.repo,
                                            'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                                            dir=ana_dir)


t_max = ana11.mjd_max
t_min = ana11.mjd_min

sources = src_load()

# Check if sources are inside the analysis time frame
srcs_all = [src for src in sources if src["mjd"] <= t_max and src["mjd"] >= t_min]

if len(srcs_all) < len(sources):
    print("Number of possible sources reduced ({} -> {}) due to analysis time frame".format(len(sources),len(srcs_all)))
    srcs = srcs_all

n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_all = [src["signal"] for src in sources]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)
signals_mask_2 = np.in1d(signals_all,signals_used)

src_id_all = np.reshape(np.argwhere(signals_mask_2 == True), n_srcs)


def plot_gamma_auto_for_thesis_time_dep(_src_id_,name,overbound=0.1, fontsize=12):
        gamma_len = len(gamma_list)
        all_x = np.concatenate(all_ns_used)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        label_x = xmin + 2*overbound*xlength
        ymax = np.amax(np.concatenate([np.array(_src[0]) + np.array(_src[1]) for _src in all_time_dep_gamma]))
        ymin = np.amin(np.concatenate([np.array(_src[0]) - np.array(_src[1]) for _src in all_time_dep_gamma]))
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        fig, axs = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(11,10))
        axs = axs.ravel()
        for i,_src_id in enumerate(_src_id_):
                x = all_ns_used[i]
                y = all_time_dep_gamma[i][0]
                yerr = all_time_dep_gamma[i][1]
                ax = axs[i]
                ax.axhline(2, ls="--", color='black',label=r'$\gamma={}$'.format(2))
                ax.errorbar(x, y, yerr=yerr, fmt='.',label=r'$\hat\gamma$')
                ax.text(label_x,label_y,r'Nr. ${}$'.format(_src_id + 1),fontsize=fontsize)
                ax.grid('on', linestyle='--', alpha=0.5)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                #ax.legend(loc='best')
                if(i==9 or i==8 or i==7 or i==6):
                        ax.set_xlabel(r'$n_{inj}$')
                        plt.setp(ax.get_xticklabels(), visible=True)
                else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                if(i==0 or i==4 or i==8):
                        ax.set_ylabel(r'$\hat\gamma$')
                else:
                        plt.setp(ax.get_yticklabels(), visible=False)
                for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        axs[-1].set_visible(False)
        axs[-2].plot(15,2,"--",markersize = .1,c="black",label=r"$\gamma=2$") #for the label
        axs[-2].errorbar([15],[2],yerr=[0],fmt='.',label=r"$\hat{\gamma}$")
        axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
        axs[-2].set_frame_on(False)
        axs[-2].set_xticks([])
        axs[-2].set_yticks([])
        fig.suptitle(r'set $\gamma$ vs fitted $\hat\gamma$', fontsize=fontsize+2)
        fig.tight_layout()
        plt.savefig(name)
        plt.clf()
        return

def plot_ns_auto_for_thesis_time_dep(_src_id_,name,overbound=0.1, fontsize=12):
        gamma_len = len(gamma_list)
        all_x = np.concatenate(all_ns_used)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        label_x = xmin + 2*overbound*xlength
        ymax = np.amax(np.concatenate([np.array(_src[0]) + np.array(_src[1]) for _src in all_time_dep_ns]))
        ymin = np.amin(np.concatenate([np.array(_src[0]) - np.array(_src[1]) for _src in all_time_dep_ns]))
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        fig, axs = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(11,10))
        axs = axs.ravel()
        for i,_src_id in enumerate(_src_id_):
                x = all_ns_used[i]
                y = all_time_dep_ns[i][0]
                yerr = all_time_dep_ns[i][1]
                ax = axs[i]
                _x = np.linspace(0,np.amax(x),1000)
                ax.plot(_x,_x, ls="--", color='black',label=r'$n_{\mathrm{inj}}=\hat{n}_{\mathrm{S}}$')
                ax.errorbar(x, y, yerr=yerr, fmt='.',label=r'$\hat{n}_\mathrm{S}$')
                ax.text(label_x,label_y,r'Nr. ${}$'.format(_src_id + 1),fontsize=fontsize)
                ax.grid('on', linestyle='--', alpha=0.5)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                #ax.legend(loc='best')
                if(i==9 or i==8 or i==7 or i==6):
                        ax.set_xlabel(r'$n_\mathrm{inj}$')
                        plt.setp(ax.get_xticklabels(), visible=True)
                else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                if(i==0 or i==4 or i==8):
                        ax.set_ylabel(r'$\hat{n}_\mathrm{S}$')
                else:
                        plt.setp(ax.get_yticklabels(), visible=False)
                for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        axs[-1].set_visible(False)
        axs[-2].plot(15,15,"--",markersize = .1,c="black",label=r"$n_{\mathrm{inj}}=\hat{n}_{\mathrm{S}}$") #for the label
        axs[-2].errorbar([15],[15],yerr=[0],fmt='.',label=r"$\hat{n}_{\mathrm{S}}$")
        axs[-2].legend(loc="center",edgecolor="white",prop={'size': 15},framealpha=1)
        axs[-2].set_frame_on(False)
        axs[-2].set_xticks([])
        axs[-2].set_yticks([])
        fig.suptitle(r'set $n_{\mathrm{inj}}$ vs fitted $\hat{n}_{\mathrm{S}}$', fontsize=fontsize+2)
        fig.tight_layout()
        plt.savefig(name)
        plt.clf()
        return


def plot_ns_gamma_auto_for_thesis_time_dep(name,overbound=0.1, fontsize=12):
        gamma_len = len(gamma_list)
        all_x = np.concatenate(all_ns_used)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        label_x = xmin + 2*overbound*xlength
        ymax = np.amax(np.concatenate([np.array(_src[0]) + np.array(_src[1]) for _src in all_time_dep_ns]))
        ymin = np.amin(np.concatenate([np.array(_src[0]) - np.array(_src[1]) for _src in all_time_dep_ns]))
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8,4))
        axs = axs.ravel()
        x = all_ns_used[0]
        y = all_time_dep_ns[0][0]
        yerr = all_time_dep_ns[0][1]
        ax = axs[0]
        _x = np.linspace(0,np.amax(x),1000)
        ax.plot(_x,_x, ls="--", color='black',label=r'$n_{\mathrm{inj}}=\hat{n}_{\mathrm{S}}$')
        ax.errorbar(x, y, yerr=yerr, fmt='.',label=r'$\hat{n}_\mathrm{S}$')
        plt.suptitle(r'Source Nr. ${}$'.format(src_id_all[0] + 1),fontsize=fontsize)
        ax.grid('on', linestyle='--', alpha=0.5)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.legend(loc='upper left')
        ax.set_title(r'set $n_{\mathrm{inj}}$ vs fitted $\hat{n}_{\mathrm{S}}$')
        ax.set_xlabel(r'$n_\mathrm{inj}$')
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.set_ylabel(r'$\hat{n}_\mathrm{S}$')
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        gamma_len = len(gamma_list)
        all_x = np.concatenate(all_ns_used)
        xmin = np.amin(all_x)
        xmax = np.amax(all_x)
        xlength = xmax-xmin
        xmin-=overbound*xlength
        xmax+=overbound*xlength
        label_x = xmin + 2*overbound*xlength
        ymax = np.amax(np.concatenate([np.array(_src[0]) + np.array(_src[1]) for _src in all_time_dep_gamma]))
        ymin = np.amin(np.concatenate([np.array(_src[0]) - np.array(_src[1]) for _src in all_time_dep_gamma]))
        ylength = ymax-ymin
        label_y = ymax - overbound*ylength
        ymin -= overbound*ylength
        ymax += overbound*ylength
        x = all_ns_used[0]
        y = all_time_dep_gamma[0][0]
        yerr = all_time_dep_gamma[0][1]
        ax = axs[1]
        ax.axhline(2, ls="--", color='black',label=r'$\gamma={}$'.format(2))
        ax.errorbar(x, y, yerr=yerr, fmt='.',label=r'$\hat\gamma$')
        ax.grid('on', linestyle='--', alpha=0.5)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel(r'$n_{inj}$')
        ax.set_title(r'set $\gamma$ vs fitted $\hat\gamma$')
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.set_ylabel(r'$\hat\gamma$')
        ax.legend(loc='best')
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fontsize)
        fig.tight_layout()
        plt.savefig(name)
        plt.clf()
        return


print("plotting...")


plot_gamma_auto_for_thesis_time_dep(src_id_all,"test_plots/gamma_fit_time_dep.pdf")
plot_ns_auto_for_thesis_time_dep(src_id_all,"test_plots/ns_fit_time_dep.pdf")
plot_ns_gamma_auto_for_thesis_time_dep("test_plots/ns_gamma_fit_time_dep_1.pdf")
