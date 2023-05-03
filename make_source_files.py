# coding: utf-8

"""
Use runlists to build a JSON file with the needed source information and to
which sample each on belongs to.
"""

import os
import json
from glob import glob
import gzip
import numpy as np
import astropy.time as astrotime

from _paths import PATHS
#from _loader import runlist_loader

def create_source_rec(source_files):
	_sources = []
	for src_file in source_files:
		with gzip.open(src_file) as _f:
			src_dict = json.load(_f)
		# Build a compact version with all relevant infos
		src_i = {}
		for key in ["run_id", "event_id", "mjd"]:
			src_i[key] = src_dict[key]
		# Store best fit from direct local trafo and map maximum
		src_i["ra"] = src_dict["bf_equ"]["ra"]
		src_i["dec"] = src_dict["bf_equ"]["dec"]
		src_i["ra_map"] = src_dict["bf_equ_pix"]["ra"]
		src_i["dec_map"] = src_dict["bf_equ_pix"]["dec"]
		# Also store the path to the original file which contains the skymap
		src_i["map_path"] = src_file
		_sources.append(src_i)
		print("Loaded EHE source from run {}:\n  {}".format(
			src_i["run_id"], src_file))
	return _sources

ehe_src_path = os.path.join(PATHS.local, "ehe_scan_maps_truncated")
hese_src_path = os.path.join(PATHS.local, "hese_scan_maps_truncated")
#runlist_path = os.path.join(PATHS.local, "runlists")

outpath = os.path.join(PATHS.local, "source_list")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Load runlists
#runlists = runlist_loader("all")

# Load sources up to EHE 6yr, list from:
#   https://wiki.icecube.wisc.edu/index.php/Analysis_of_pre-public_alert_HESE/EHE_events#EHE
# Last Run ID is 127910 from late 86V (2015) run, next from 7yr is 128290
ehe_src_files = sorted(glob(os.path.join(ehe_src_path, "*.json.gz")))
hese_src_files = sorted(glob(os.path.join(hese_src_path, "*.json.gz")))

ehe_sources = np.array(create_source_rec(ehe_src_files))
hese_sources = np.array(create_source_rec(hese_src_files))

print("Number of considered ehe sources: {}".format(len(ehe_src_files)))
print("Number of considered hese sources: {}".format(len(hese_src_files)))

# since we are using ps_tracks_version-004-p00
datasets = ['IC79','IC86_2011','IC86_2012','IC86_2013','IC86_2014','IC86_2015','IC86_2016','IC86_2017','IC86_2018','IC86_2019']
version = 'version-004-p00'
data_path = os.path.join('/data','ana','analyses','ps_tracks',version)

# Match sources with their seasons and store to JSON
# We could just copy the list from the wiki here, but let's just check with
# the runlists
sources_per_sam = {}
src_t = np.array([src_["mjd"] for src_ in ehe_sources])
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
    sources_per_sam[name] = ehe_sources[t_mask].tolist()
    print("  {} sources in this sample.".format(np.sum(t_mask)))
assert sum(map(len, sources_per_sam.values())) == len(ehe_sources)

# cant use above method for hese, cause samples are overlapping
# doing it manually instead

src_t = np.array([src_["mjd"] for src_ in hese_sources])

sources_per_sam['IC79'].extend(hese_sources[:3].tolist())
sources_per_sam['IC86_2011'].extend(hese_sources[3:7].tolist())
sources_per_sam['IC86_2012'].extend(hese_sources[7:8].tolist())
sources_per_sam['IC86_2013'].extend(hese_sources[8:14].tolist())
sources_per_sam['IC86_2014'].extend(hese_sources[14:19].tolist())
sources_per_sam['IC86_2015'].extend(hese_sources[19:].tolist())

assert sum(map(len, sources_per_sam.values())) == (len(ehe_sources)+len(hese_sources))

with open(os.path.join(outpath, "source_list.json"), "w") as outf:
    json.dump(sources_per_sam, fp=outf, indent=2)
