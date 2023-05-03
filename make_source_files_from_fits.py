# create source files from fits using cut on i3type and signalness

import os
import json
from glob import glob
import gzip
import numpy as np
import healpy as hp
import histlite as hl

import matplotlib.pyplot as plt

import csky as cy

from _paths import PATHS

import math
## Show skymap
def draw_skymap(skymap,name="testmap.pdf"):
	hp.mollview(skymap, title="", cbar=True, notext=True, hold=False)
	hp.graticule()
	plt.text(2.0, 0., r"$0^\circ$", ha="left", va="center")
	plt.text(1.9,0.45, r"$30^\circ$", ha="left", va="center")
	plt.text(1.4,0.8, r"$60^\circ$", ha="left", va="center")
	plt.text(1.9,-0.45, r"$-30^\circ$", ha="left", va="center")
	plt.text(1.4,-0.8, r"$-60^\circ$", ha="left", va="center")
	plt.text(2.0, -0.15, r"$180^\circ$", ha="center", va="center")
	plt.text(1.333, -0.15, r"$240^\circ$", ha="center", va="center")
	plt.text(.666, -0.15, r"$300^\circ$", ha="center", va="center")
	plt.text(0.0, -0.15, r"$0^\circ$", ha="center", va="center")
	plt.text(-.666, -0.15, r"$60^\circ$", ha="center", va="center")
	plt.text(-1.333, -0.15, r"$120^\circ$", ha="center", va="center")
	plt.text(-2.0, -0.15, r"$180^\circ$", ha="center", va="center")
	#plt.draw()
	plt.savefig(name)

def create_source_rec(i3type, fits_path, signal = 0.):
        fit_files = list(glob(os.path.join(fits_path,'*.*')))
        _sources = []
        file_count = len(fit_files)
        mom_count = 0
        for src_file in fit_files:
                mom_count+=1
                print('\r', str(mom_count), end = ' of {} ({}%)'.format(file_count,int(mom_count/file_count * 100)))
                skymap, header = hp.read_map(src_file,h=True, verbose=False)
                header = dict(header)
                _signal = header['SIGNAL']
                _i3type = header['I3TYPE']
                #print(_signal)
                #print(_i3type, i3type)
                if (_i3type == i3type) and (_signal >= signal):
                        # Build a compact version with all relevant infos
                        src_i = {}
                        src_i["run_id"] = header['RUNID']
                        src_i["event_id"] = header['EVENTID']
                        src_i["mjd"] = header['EVENTMJD']
                        src_i["ra"] = 2*np.pi * header["RA"]/360.
                        src_i["dec"] = 2*np.pi * header["DEC"]/360.
                        src_i["ra_deg"] = header["RA"]
                        src_i["dec_deg"] = header["DEC"]
                        src_i["signal"] = _signal
                        src_i["energy"] = header["ENERGY"]
                        # Also store the path to the original file which contains the skymap
                        src_i["map_path"] = src_file
                        _sources.append(src_i)
                        print("")
                        print(_i3type)
                        print("")
                        #from IPython import embed
                        #embed()
                        print("Loaded source from run {}:\n  {}".format(
                                src_i["run_id"], src_file))
        print("")
        print("Found {} sources of type {}".format(len(_sources),i3type))
        return _sources

path_to_fits = os.path.join("/data", "ana", "realtime", "alert_catalog_v2", "fits_files")


src_type = 'gfu-gold'

srcs = create_source_rec(src_type, path_to_fits)

# all golds for test purposes, not fast but anyways...
all_golds = []
all_golds.extend(srcs)
ehe_gold = create_source_rec('ehe-gold', path_to_fits)
hese_gold = create_source_rec('hese-gold', path_to_fits)
all_golds.extend(ehe_gold)
all_golds.extend(hese_gold)


outpath = os.path.join(PATHS.local, "source_list_from_fits")
if not os.path.isdir(outpath):
    print("Creating outpath {}".format(outpath))
    os.makedirs(outpath)

print("Saving source_list.json at {}".format(outpath))
with open(os.path.join(outpath, "source_list.json"), "w") as outf:
    json.dump(srcs, fp=outf, indent=2)

print("Saving all_golds.json at {}".format(outpath))
with open(os.path.join(outpath, "all_golds.json"), "w") as outf:
    json.dump(all_golds, fp=outf, indent=2)


"""
from IPython import embed
embed()

# since we are using ps_tracks_version-004-p00
datasets = ['IC79','IC86_2011','IC86_2012','IC86_2013','IC86_2014','IC86_2015','IC86_2016','IC86_2017','IC86_2018','IC86_2019']
version = 'version-004-p00'
data_path = os.path.join('/data','ana','analyses','ps_tracks',version)
"""
"""
# Match sources with their seasons and store to JSON
# We could just copy the list from the wiki here, but let's just check with
# the runlists
sources_per_sam = {}
src_t = np.array([src_["mjd"] for src_ in srcs])
for name in datasets:
    data = np.load(os.path.join(data_path, name + '_exp.npy'))
    #from IPython import embed
    #embed()

    print("Match sources for sample {}".format(name))
    tmin = np.amin(data['time'])
    tmax = np.amax(data['time'])
    print(name,tmin,tmax)
    t_mask = (src_t >= tmin) & (src_t <= tmax)
    # Store all sources for the current sample
    sources_per_sam[name] = srcs[t_mask].tolist()
    print("  {} sources in this sample.".format(np.sum(t_mask)))
assert sum(map(len, sources_per_sam.values())) == len(srcs)

outpath = os.path.join(PATHS.local, "source_list_from_fits")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

with open(os.path.join(outpath, "source_list.json"), "w") as outf:
    json.dump(sources_per_sam, fp=outf, indent=2)
"""
"""


decs = [sr["dec"] for sr in srcs]
ras = [sr["ra"] for sr in srcs]

src = cy.utils.Sources(ra = ras, dec = decs)

# setting resolution down for bigger points
# better method would be getting pixel coordinates of matrix entries
# and draw data point afterwards
src_map = hl.heal.hist(80, src.dec, src.ra)#512

print("src_map")
from IPython import embed
embed()


fig, ax = plt.subplots (subplot_kw=dict (projection='aitoff'))
sp = cy.plotting.SkyPlotter(pc_kw=dict(cmap='Greys', vmin=0))
mesh, cb = sp.plot_map(ax, np.where(src_map.map>0, src_map.map, np.nan), n_ticks=2)
print("mesh")
from IPython import embed
embed()
kw = dict(color='.5', alpha=.5)
#ax.plot(ras,decs,".",markersize=10,c="blue")
print("sp")
from IPython import embed
embed()
#sp.plot(ras,decs,".",markersize=10,c="red")
sp.plot_gp(ax, lw=.5, **kw)
sp.plot_gc(ax, **kw)
ax.grid(**kw)
ax.scatter(ras,decs,color="red",s=10)
x,y = sp.thetaphi_to_mpl(ras,decs)
ax.scatter(x,y,color="blue",s=10)
plt.tight_layout()
plt.savefig("test_plots/gfu_gold_skymap.pdf")

"""
"""
fit_file = list(glob(os.path.join(path_to_fits,'*.*')))
src_file = fit_file[0]
skymap, header = hp.read_map(src_file,h=True, verbose=False)

draw_skymap(skymap,name="test_plots/contour_test_1.pdf")
## Get 50% contour
quantile = 0.50 # to get 50% contour
header = dict(header)
NSIDE = header['NSIDE']
argsort = np.argsort(-skymap)
cum_skymap = np.cumsum(sorted(skymap,reverse=True))
cont_ind = argsort[cum_skymap < quantile]
contour = np.array([1. if pix in cont_ind else 0. for pix in range(len(skymap))])
from IPython import embed
embed()
draw_skymap(contour, name = "test_plots/contour_test.pdf")
"""
