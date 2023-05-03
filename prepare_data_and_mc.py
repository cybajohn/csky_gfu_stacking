
"""
Remove ehe and hese from mc and used ehe and hese sources from data
"""

import os
import json
import gzip
import numpy as np

from _paths import PATHS
from _loader import source_list_loader
import csky as cy

import matplotlib.pyplot as plt

"""

def remove_ehe_hese_from_mc(mc, eheids, heseids): # or rather identify them
    
    Mask all values in ``mc`` that have the same run and event ID combination
    as in ``eheids`` and ``heseids``.

    Parameters
    ----------
    mc : record-array
        MC data, needs names ``'Run', 'Event'``.
    eheids : dict or record-array
        Needs names / keys ``'run_id', 'event_id``.

    Returns
    -------
    is_ehe_like : array-like, shape (len(mc),)
        Mask: ``True`` for each event in ``mc`` that is EHE like.
    
    # Make combined IDs to easily match against EHE IDs with `np.isin`
    factor_mc = 10**np.ceil(np.log10(np.amax(mc["Event"])))
    _evids = np.atleast_1d(eheids["event_id"])
    factor_ehe = 10**np.ceil(np.log10(np.amax(_evids)))
    factor = max(factor_mc, factor_ehe)

    combined_mcids = (factor * mc["Run"] + mc["Event"]).astype(int)
    assert np.all(combined_mcids > factor)  # Is int overflow a thing here?

    _runids = np.atleast_1d(eheids["run_id"])
    combined_eheids = (factor * _runids + _evids).astype(int)
    assert np.all(combined_eheids > factor)

    # Check which MC event is tagged as EHE like
    is_ehe_like = np.in1d(combined_mcids, combined_eheids)
    print("  Found {} / {} EHE like events in MC".format(np.sum(is_ehe_like),
                                                          len(mc)))
    return is_ehe_like

"""

def remove_ehe_from_mc(mc, eheids): # or rather identify them
    """
    Mask all values in ``mc`` that have the same run and event ID combination
    as in ``eheids``.

    Parameters
    ----------
    mc : record-array
        MC data, needs names ``'Run', 'Event'``.
    eheids : dict or record-array
        Needs names / keys ``'run_id', 'event_id``.

    Returns
    -------
    is_ehe_like : array-like, shape (len(mc),)
        Mask: ``True`` for each event in ``mc`` that is EHE like.
    """
    # Make combined IDs to easily match against EHE IDs with `np.isin`
    factor_mc = 10**np.ceil(np.log10(np.amax(mc["Event"])))
    _evids = np.atleast_1d(eheids["event_id"])
    factor_ehe = 10**np.ceil(np.log10(np.amax(_evids)))
    factor = max(factor_mc, factor_ehe)

    combined_mcids = (factor * mc["Run"] + mc["Event"]).astype(int)
    assert np.all(combined_mcids > factor)  # Is int overflow a thing here?

    _runids = np.atleast_1d(eheids["run_id"])
    combined_eheids = (factor * _runids + _evids).astype(int)
    assert np.all(combined_eheids > factor)

    # Check which MC event is tagged as EHE like
    is_ehe_like = np.in1d(combined_mcids, combined_eheids)
    print("  Found {} / {} EHE like events in MC".format(np.sum(is_ehe_like),
                                                          len(mc)))
    return is_ehe_like


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

# remove srcs from data

names = source_list_loader()

#names = names[1:-1]

data_path = os.path.join("/data", "ana", "analyses", "ps_tracks", "version-004-p00")
out_dir = os.path.join(path.data, "cleared_data_and_mc")

if not os.path.isdir(out_dir):
	print("{} does not exist... creating it".format(out_dir))
	os.makedirs(out_dir)

for name in names:
        srcs = source_list_loader(name)
        src_id = [src["event_id"] for src in srcs[name]]
	data = np.load(os.path.join(data_path,str(name)+"_exp.npy"))
	inv_mask = np.in1d(data["event"],src_id)
	data_without_src = data[~inv_mask]
	np.save(data_without_src, out_dir+str(name)+"_exp.npy")	




files = "../../../../data/ana/analyses/ps_tracks/version-004-p00/IC86_2016_MC.npy"

mc = np.load(files)

with gzip.open("out_test/check_ehe_alert_mc_ids/IC86_2016.json.gz") as json_file:
	ehealertids = json.load(json_file)

with gzip.open("out_test/check_ehe_mc_ids/IC86_2012.json.gz") as json_file:
        eheids = json.load(json_file)

with gzip.open("out_test/check_my_ehe_mc_ids/IC86_2012.json.gz") as json_file:
        myeheids = json.load(json_file)

with gzip.open("out_test/check_hese_mc_ids/IC86_2016.json.gz") as json_file:
        heseids = json.load(json_file)

with gzip.open("out_test/check_ehe_alert_hb_mc_ids/IC86_2016.json.gz") as json_file:
        ehealerthbids = json.load(json_file)

with gzip.open("out_test/check_ehe_alert_mc_ids_bf/IC86_2016.json.gz") as json_file:
        ehealertidsbf = json.load(json_file)

with gzip.open("out_test/check_hese_mc_ids_bf/IC86_2016.json.gz") as json_file:
        heseidsbf = json.load(json_file)


ehefilter = np.in1d(mc['event'],eheids['event_id'])
myehefilter = np.in1d(mc['event'],myeheids['event_id'])
ehealertfilter = np.in1d(mc['event'],ehealertids['event_id'])
hesefilter = np.in1d(mc['event'],heseids['event_id'])
ehealerthbfilter = np.in1d(mc['event'],ehealerthbids['event_id'])

hesefilterbf = np.in1d(mc['event'],heseidsbf['event_id'])
ehealertfilterbf = np.in1d(mc['event'],ehealertidsbf['event_id'])

hese_ehe = ehealertfilter + hesefilter

hese_ehe_bf = ehealertfilterbf + hesefilterbf

#from IPython import embed
#embed()


_bins = np.linspace(-1,1,50)

print("plotting...")
_log = True
#plt.hist(np.sin(mc[~ehefilter]['trueDec']),weights=mc[~ehefilter]['ow']*fluss(mc[~ehefilter]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='ehe')
print("done 1 plot...")
plt.hist(np.sin(mc['trueDec']),weights=mc['ow']*fluss(mc['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc')
#plt.hist(np.sin(mc[~myehefilter]['trueDec']),weights=mc[~myehefilter]['ow']*fluss(mc[~myehefilter]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='myehe')
plt.hist(np.sin(mc[~ehealertfilter]['trueDec']),weights=mc[~ehealertfilter]['ow']*fluss(mc[~ehealertfilter]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o EHEAlertFilter_15')
plt.hist(np.sin(mc[~ehealerthbfilter]['trueDec']),weights=mc[~ehealerthbfilter]['ow']*fluss(mc[~ehealerthbfilter]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o EHEAlertFilterHB_15')
plt.hist(np.sin(mc[~hesefilter]['trueDec']),weights=mc[~hesefilter]['ow']*fluss(mc[~hesefilter]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o HESEFilter_15')
plt.hist(np.sin(mc[~hese_ehe]['trueDec']),weights=mc[~hese_ehe]['ow']*fluss(mc[~hese_ehe]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o HESE and EHEAlert')

plt.hist(np.sin(mc[~ehealertfilterbf]['trueDec']),weights=mc[~ehealertfilterbf]['ow']*fluss(mc[~ehealertfilterbf]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o EHEAlertFilter_15_bf')
plt.hist(np.sin(mc[~hesefilterbf]['trueDec']),weights=mc[~hesefilterbf]['ow']*fluss(mc[~hesefilterbf]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o HESEFilter_15_bf')
plt.hist(np.sin(mc[~hese_ehe_bf]['trueDec']),weights=mc[~hese_ehe_bf]['ow']*fluss(mc[~hese_ehe_bf]['trueE'],1,1,2),log=_log,bins=_bins,histtype='step',label='mc w/o HESE_bf and EHEAlert_bf')

plt.xlabel('sindec')
plt.legend(loc='best')
plt.savefig('test_plots/mc/mc_test_004_p00_bf.pdf')


#from IPython import embed
#embed()

