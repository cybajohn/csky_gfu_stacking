"""
combine gold filter arrays
"""

import sys
import os
import json
from glob import glob
import gzip
import numpy as np

from IPython import embed

import matplotlib.pyplot as plt

from _paths import PATHS


inpath = os.path.join(PATHS.data, "check_gold_mc_ids_new_gcd")
outpath = os.path.join(PATHS.local, "check_gold_mc_ids_new_gcd")


npy_files = list(glob(os.path.join(inpath,'*.*')))

file_count = len(npy_files)

run_id = []
event_id = []
prim_energy = []
ow = []
azi = []
zen = []

for i,_path in enumerate(npy_files):
    print('\r', str(i+1), end = ' of {}'.format(file_count))	
    _file = np.load(_path)
    gfu_gold = _file['PassGFUGold']
    _azi = _file["PrimAzimuth"]
    _zen = _file["PrimZenith"]
    _prim_energy = _file['PrimEnergy']
    _ow = _file['OneWeight']
    run_ids = _file['RunID']
    event_ids = _file['EventID']
    run_id.extend(run_ids[gfu_gold])
    event_id.extend(event_ids[gfu_gold])
    azi.extend(_azi[gfu_gold])
    zen.extend(_zen[gfu_gold])
    prim_energy.extend(_prim_energy[gfu_gold])
    ow.extend(_ow[gfu_gold])

print("")

out_dict = {"gold_run_id": run_id, "gold_event_id": event_id, "azi": azi, "zen": zen, 'ow': ow, 'prim_energy': prim_energy}

run_id = []
event_id = []
prim_energy = []
ow = []

for i,_path in enumerate(npy_files):
        print('\r', str(i+1), end = ' of {}'.format(file_count))
        _file = np.load(_path)
        ehe_gold = _file['PassEHEGold']
        _prim_energy = _file['PrimEnergy']
        _ow = _file['OneWeight']
        run_ids = _file['RunID']
        event_ids = _file['EventID']
        run_id.extend(run_ids[ehe_gold])
        event_id.extend(event_ids[ehe_gold])
        prim_energy.extend(_prim_energy[ehe_gold])
        ow.extend(_ow[ehe_gold])
        

print("")

out_dict2 = {"gold_run_id": run_id, "gold_event_id": event_id, 'ow': ow, 'prim_energy': prim_energy}

run_id = []
event_id = []

for i,_path in enumerate(npy_files):
        print('\r', str(i+1), end = ' of {}'.format(file_count))
        _file = np.load(_path)
        hese_gold = _file['PassHESEGold']
        run_ids = _file['RunID']
        event_ids = _file['EventID']
        run_id.extend(run_ids[hese_gold])
        event_id.extend(event_ids[hese_gold])

print("")

out_dict3 = {"gold_run_id": run_id, "gold_event_id": event_id}



files = "../../../../data/ana/analyses/ps_tracks/version-004-p00/IC86_2016_MC.npy"

mc = np.load(files)

mc_combined = mc['event'] + 1e10 * mc['run']
mine_combined = np.array(out_dict["gold_event_id"]) + 1e10 * np.array(out_dict["gold_run_id"])

mine_combined2 = np.array(out_dict2["gold_event_id"]) + 1e10 * np.array(out_dict2["gold_run_id"])
mine_combined3 = np.array(out_dict3["gold_event_id"]) + 1e10 * np.array(out_dict3["gold_run_id"])


mask = np.in1d(mc_combined,mine_combined)

mask2 = np.in1d(mc_combined,mine_combined2)
mask3 = np.in1d(mc_combined,mine_combined3)

mc_clean = mc[~mask]

mc_clean2 = mc[~mask2]
mc_clean3 = mc[~mask3]


mc_ehe_gfu = mc[mask]
print("check mc")
embed()
test = np.in1d(mc["trueAzi"],out_dict["azi"])
test2 = np.in1d(mc["trueZen"],out_dict["zen"])
test3 = np.in1d(mc["trueE"],out_dict["prim_energy"])
test4 = test*test2*test3
mc_ehe_gfu = mc[test4]

embed()

version = "version-004-p00"

grl_inpath = "../../../../data/ana/analyses/ps_tracks/{}/GRL".format(version)

# only IC86
grl_npy_files = list(glob(os.path.join(grl_inpath,'IC86_*_exp.npy')))

livetimes = 0

for data_file in grl_npy_files:
    file_name = os.path.basename(data_file)
    print("Working with {}".format(file_name))
    data = np.load(data_file)
    livetimes += np.sum(data['livetime'])

livetimes = livetimes * 24. * 60. * 60.

def mc_weight(mc, live):
    return 1.36*mc['ow']*(live*((mc['trueE']/1e5)**(-2.37))*(10**(-18)))

values = plt.hist(np.sin(mc_ehe_gfu['dec']),weights=mc_weight(mc_ehe_gfu,livetimes))
plt.xlabel(r"$\sin{(\mathrm{dec})}$")
plt.ylabel(r"number of events")
plt.savefig("test_plots/mc_gold_sindec_hist.pdf")
plt.clf()


mc_band1 = mc_ehe_gfu[(mc_ehe_gfu["trueDec"]<(-2*np.pi*5/360))*(mc_ehe_gfu["trueDec"]>=(-2*np.pi*90/360))]
mc_band2 = mc_ehe_gfu[(mc_ehe_gfu["trueDec"]<(2*np.pi*30/360))*(mc_ehe_gfu["trueDec"]>=(-2*np.pi*5/360))]
mc_band3 = mc_ehe_gfu[(mc_ehe_gfu["trueDec"]<(2*np.pi*90/360))*(mc_ehe_gfu["trueDec"]>=(2*np.pi*30/360))]

print("plot effective_area")

log_bins1 = np.logspace(1,6,30)

hist1, bins1, _ = plt.hist(mc_band1["trueE"], weights = mc_band1["ow"], bins = log_bins1)
plt.clf()
bin_lengths = bins1[1:] - bins1[:-1]
hist1 = hist1/bin_lengths / 85
plt.hist(bins1[:-1], bins1, weights=hist1, histtype="step", label = r"$\delta \in [-90\degree,-5\degree]$")
plt.xscale("log")
plt.yscale("log")
#plt.xlim(10,10**6)
plt.ylim(10**(-5),10**7)
plt.xlabel("trueE in TeV")
plt.ylabel("effective area in m$^2$")
plt.legend(loc="best")
plt.savefig("test_plots/effective_area_1.pdf")
plt.clf()

hist2, bins2, _ = plt.hist(mc_band2["trueE"], weights = mc_band2["ow"], bins = log_bins1)
plt.clf()
bin_lengths = bins2[1:] - bins2[:-1]
hist2 = hist2/bin_lengths / 35
plt.hist(bins2[:-1], bins2, weights=hist2, histtype="step", label = r"$\delta \in [-5\degree,30\degree]$")
plt.xscale("log")
plt.yscale("log")
#plt.xlim(10,10**6)
plt.ylim(10**(-5),10**7)
plt.xlabel("trueE in TeV")
plt.ylabel("effective area in m$^2$")
plt.legend(loc="best")
plt.savefig("test_plots/effective_area_2.pdf")
plt.clf()


hist3, bins3, _ = plt.hist(mc_band3["trueE"], weights = mc_band3["ow"], bins = log_bins1)
plt.clf()
bin_lengths = bins3[1:] - bins3[:-1]
hist3 = hist3/bin_lengths / 60
plt.hist(bins3[:-1], bins3, weights=hist3, histtype="step", label = r"$\delta \in [30\degree,90\degree]$")
plt.xscale("log")
plt.yscale("log")
#plt.xlim(10,10**6)
plt.ylim(10**(-5),10**7)
plt.xlabel("trueE in TeV")
plt.ylabel("effective area in m$^2$")
plt.legend(loc="best")
plt.savefig("test_plots/effective_area_3.pdf")
plt.clf()

embed()

def fluss(trueE, phi0, E0, gamma):
        """
        calculates the flux

        Parameters
        ----------
        mc_sample: dict
                mc sample
        phi0: double
                norm of flux
        E0: double
                energy
        gamma: double
                gamma

        Return
        ------
        flux: double
                flux
        """
        return phi0*(trueE/E0)**(-gamma)


from IPython import embed
embed()

#plt.hist2d(np.sin(mc['trueDec']),np.log10(mc['trueE']),weights=mc['ow']*(mc['trueE'])**(-2), norm=LogNorm(), bins=20)

_bins = np.linspace(-1,1,50)
_log = True
plt.hist(np.sin(mc_clean['trueDec']),weights=mc_clean['ow']*fluss(mc_clean['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o gfu_gold')
plt.hist(np.sin(mc_clean2['trueDec']),weights=mc_clean2['ow']*fluss(mc_clean2['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o ehe_gold')
plt.hist(np.sin(mc_clean3['trueDec']),weights=mc_clean3['ow']*fluss(mc_clean3['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o hese_gold')
plt.hist(np.sin(mc['trueDec']),weights=mc['ow']*fluss(mc['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc')
plt.xlabel('sindec')
plt.legend(loc='best')
plt.savefig('test_plots/mc/mc_gfu_gold_log.pdf')
plt.clf()

_log = False
plt.hist(np.sin(mc_clean['trueDec']),weights=mc_clean['ow']*fluss(mc_clean['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o gfu_gold')
plt.hist(np.sin(mc_clean2['trueDec']),weights=mc_clean2['ow']*fluss(mc_clean2['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o ehe_gold')
plt.hist(np.sin(mc_clean3['trueDec']),weights=mc_clean3['ow']*fluss(mc_clean3['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o hese_gold')
plt.hist(np.sin(mc['trueDec']),weights=mc['ow']*fluss(mc['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc')
plt.xlabel('sindec')
plt.legend(loc='best')
plt.savefig('test_plots/mc/mc_gfu_gold_non_log.pdf')
plt.clf()

from IPython import embed
embed()
