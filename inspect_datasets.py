"""
To describe the used datasets
"""

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl
import os
from glob import glob
import csky as cy

from _loader import easy_source_list_loader as src_load
from _paths import PATHS


version = "version-004-p00"

grl_inpath = "../../../../data/ana/analyses/ps_tracks/{}/GRL".format(version)

npy_files = list(glob(os.path.join(grl_inpath,'*_exp.npy')))

# remove IC59 IC40 IC79 cause theyre not used besides being bad

for item in npy_files:
    if "IC59" in item or "IC40" in item or "IC79" in item:
        npy_files.remove(item)

print(npy_files)

events_total = 0
livetime_total = 0

for file_name in npy_files:
    data_file = os.path.basename(file_name)
    print("Working with {}".format(data_file))
    data = np.load(file_name)
    livetime = np.sum(data["livetime"])
    print("livetime: ",livetime)
    events = np.sum(data["events"])
    print("events: ",events)
    events_total += events
    livetime_total += livetime

print(livetime_total)
print(events_total)
embed()
