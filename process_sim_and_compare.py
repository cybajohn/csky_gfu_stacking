"""
This file processes the sim files 21220 and 2 and filters
the gfu-gold events from these files out of the 2016 mc file.
These are then compared in energy and position to 9 years of 
gfu-gold alerts.
"""

from IPython import embed

import os
import sys
import numpy as np
import json
import glob
import healpy as hp

from icecube import icetray, dataclasses, dataio
from icecube.realtime_gfu.muon_alerts import gfu_alert_eval

import matplotlib.pyplot as plt

#-----------------------------------
#   define some functions
#-----------------------------------

def create_source_rec(i3type, fits_path):
    """
    Collects the real alerts and returns a dict
    with relevant information.
    -------------------------------------------
    Parameters:
        i3type: str, the alert type
        fits_path: str, path to fits files
    -------------------------------------------
    Returns:
        _sources: dict, relevant information
    """
    fit_files = list(glob.glob(os.path.join(fits_path,'*.*')))
    _sources = []
    for src_file in fit_files:
            skymap, header = hp.read_map(src_file,h=True, verbose=False)
            header = dict(header)
            _i3type = header['I3TYPE']
            if (_i3type == i3type):
                    # Build a compact version with all relevant infos
                    src_i = {}
                    src_i["mjd"] = header['EVENTMJD']
                    src_i["ra"] = 2*np.pi * header["RA"]/360.
                    src_i["dec"] = 2*np.pi * header["DEC"]/360.
                    src_i["ra_deg"] = header["RA"]
                    src_i["dec_deg"] = header["DEC"]
                    src_i["energy"] = header["ENERGY"]
                    _sources.append(src_i)
    print("")
    print("Found {} sources of type {}".format(len(_sources),i3type))
    return _sources

def gfu_gold_identifier(infiles):
    """
    Identifies the gfu-gold events and returns mc info.
    This is taken from Erik Blaufuss and shortened. 
    --------------------------------------------------
    Parameters:
        infiles: str, sim-files
    --------------------------------------------------
    Returns:
        mc_one_wt: list, weigths
        mc_prim_E: list, energy
        mc_prim_zen: list, zen
        mc_prim_azi: list, azi
    """
    events_read = 0
    mc_one_wt = []
    mc_prim_E = []
    mc_prim_zen = []
    mc_prim_azi = []
    for _file in infiles:
        afile = dataio.I3File(_file)
        while afile.more():
            events_read += 1
            pass_gfu_gold = False
            pframe = afile.pop_physics()
            if pframe.Has('AlertNamesPassed'):
                alert_list = pframe['AlertNamesPassed']
                ## Skip those boring events that don't do ANYTHING.
                if len(alert_list) == 0:
                    continue
                #if len(alert_list) > 1:
                #    print(alert_list)
                if pframe.Has('AlertShortFollowupMsg'):
                    ev_json = json.loads(pframe['AlertShortFollowupMsg'].value)
                    ## Pad the json to be like "I3Live"
                    message = {'value': { 'data' : ev_json}}
                    #print(message)            
                else:
                    if len(alert_list) > 0:
                        ## Complain only if there should be a message from any alert
                        print('Missing event dict!')
                if 'neutrino' in alert_list:
                    is_alert = gfu_alert_eval(message)
                    #if we got an alert, let's disect it
                    if is_alert['pass_tight']:
                        pass_gfu_gold = True
                        ## found a gold or bronze
                        #print("********************Found Gold or Bronze")
                        #print("GFU:",is_alert)
                        if pframe.Has('I3MCWeightDict'):
                            mc_wts = pframe['I3MCWeightDict']
                            mc_one_wt.append(mc_wts['OneWeight'])
                            mc_prim_E.append(mc_wts['PrimaryNeutrinoEnergy'])
                            mc_prim_zen.append(mc_wts['PrimaryNeutrinoZenith'])
                            mc_prim_azi.append(mc_wts['PrimaryNeutrinoAzimuth'])
                        else:
                            print("No Weight Dict!")
    return mc_one_wt, mc_prim_E, mc_prim_zen, mc_prim_azi

def mc_weight(mc, live):
    """
    mc weight: one weight * livetime * flux
    ---------------------------------------
    Parameters:
        mc: the mc file
        live: livetime in days
    ---------------------------------------
    Returns:
        mc_weight
    """
    live = live*60*60*24
    return 1.36*mc['ow']*(live*((mc['trueE']/1e5)**(-2.37))*(10**(-18)))

#-------------------------------------
#     here begins the code
#-------------------------------------


print("collect alerts from sim files")
#define paths
path_21220 = os.path.join("/data", "ana", "realtime", "alert_catalog_v2",
                    "sim_21220_alerts")
path_21002 = os.path.join("/data", "ana", "realtime", "alert_catalog_v2",
                    "sim_21002_alerts")


sim_files_21220 = sorted([f for f in glob.glob(path_21220 + "/*.i3.zst") if
                                not f.startswith("Geo")])
sim_files_21002 = sorted([f for f in glob.glob(path_21002 + "/*.i3.zst") if
                                not f.startswith("Geo")])

#collect infos to identify the gfu-gold events
#event id is the iceprod job id so that wont help

print("working with 21220...")
# would normally take all files but that takes a lot of time
# about 50 seem to already show the missmatch in energy
mc_one_wt_1, mc_prim_E_1, mc_prim_zen_1, mc_prim_azi_1 = gfu_gold_identifier(sim_files_21220[:50])
print("working with 21002...")
mc_one_wt_2, mc_prim_E_2, mc_prim_zen_2, mc_prim_azi_2 = gfu_gold_identifier(sim_files_21002[:50])

#combine
mc_one_wt = mc_one_wt_1 + mc_one_wt_2
mc_prim_E = mc_prim_E_1 + mc_prim_E_2
mc_prim_zen = mc_prim_zen_1 + mc_prim_zen_2
mc_prim_azi = mc_prim_azi_1 + mc_prim_azi_2


print("load ps_tracks version-004-p00 2016 mc and compare")

ps_tracks_mc_path = os.path.join("/data", "ana", "analyses", "ps_tracks", "version-004-p00", "IC86_2016_MC.npy")
mc = np.load(ps_tracks_mc_path)

# compare both sets in position and energy (last to be safe)
test1 = np.in1d(mc["trueAzi"],mc_prim_azi)
test2 = np.in1d(mc["trueZen"],mc_prim_zen)
test3 = np.in1d(mc["trueE"],mc_prim_E)
test4 = test1*test2*test3
mc_gfu_gold = mc[test4]

print("load real alerts and take the years 2011-2019")
# my analysis covers the years 2011 to 2019 and so does the 2016 mc file
# take only events that fall into the 9 year analysis window

t_min = 55694.419901764544
t_max = 58998.82737388495

livetimes = 3184.16 # in days

fits_path = os.path.join("/data", "ana", "realtime", "alert_catalog_v2", "fits_files") 
srcs = create_source_rec("gfu-gold", fits_path)
srcs_all = [src for src in srcs if src["mjd"] <= t_max and src["mjd"] >= t_min]

print("make comparison plot")
# this plot compares position and energy

srcs_dec = [src["dec"] for src in srcs_all]
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
axs = axs.ravel()

_bins = np.linspace(-1,1,11)
x_ticks = [-1,-0.5,0,0.5,1]

ax = axs[0]
ax.hist(np.sin(srcs_dec), bins = _bins)
ax.set_xlabel(r"$\sin{(\delta)}$",fontsize=18)
ax.set_ylabel(r"Number of events",fontsize=18)
ax.set_xticks(x_ticks)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_title("real alerts")

ax = axs[1]
ax.hist(np.sin(mc_gfu_gold['dec']),weights=mc_weight(mc_gfu_gold, livetimes), bins=_bins)
ax.set_xlabel(r"$\sin{(\delta)}$",fontsize=18)
ax.set_xticks(x_ticks)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_title("alerts in 2016 mc")

srcs_energy = [src["energy"] for src in srcs_all]

ax = axs[2]
ax.hist(srcs_energy, bins=np.logspace(1,4))
ax.set_xscale('log')
ax.set_xlabel(r"Energy in $\mathrm{TeV}$",fontsize=18)
ax.set_ylabel(r"Number of events",fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

ax = axs[3]
ax.hist((10**(mc_gfu_gold['logE']))/1e3,weights=mc_weight(mc_gfu_gold, livetimes), bins=np.logspace(1,4))
ax.set_xscale('log')
ax.set_xlabel(r"Energy in $\mathrm{TeV}$",fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("gfu_gold_mc_data_comp.pdf")
plt.clf()




