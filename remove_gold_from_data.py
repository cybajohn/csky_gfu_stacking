# remove gfu_gold from data
import csky as cy
import numpy as np
from _loader import easy_source_list_loader as src_load
from IPython import embed
from glob import glob
import os

from _paths import PATHS

sources = src_load()

run_id = [src["run_id"] for src in sources]
event_id = [src["event_id"] for src in sources]

out_dict = {"gold_run_id": run_id, "gold_event_id": event_id}

version = "version-004-p00"

inpath = "../../../../data/ana/analyses/ps_tracks/{}".format(version)
grl_inpath = "../../../../data/ana/analyses/ps_tracks/{}/GRL".format(version)

npy_files = list(glob(os.path.join(inpath,'*_exp.npy')))
grl_npy_files = list(glob(os.path.join(grl_inpath,'*_exp.npy')))

file_count = len(npy_files)

print("Found {} files in {}".format(file_count,version))
print(npy_files)

outpath = os.path.join(PATHS.data, "cleaned_datasets_new")
if os.path.isdir(outpath):
    print("Output folder '{}' is already ".format(outpath) +
                    "existing")
else:
    os.makedirs(outpath)
    print("Created outpath {}".format(outpath))

grl_outpath = os.path.join(outpath, "GRL")
if os.path.isdir(grl_outpath):
    print("Output folder '{}' is already ".format(grl_outpath) +
                    "existing")
else:
    os.makedirs(grl_outpath)
    print("Created outpath {}".format(grl_outpath))

print("Cleaning datasets")
for data_file in npy_files:
    file_name = os.path.basename(data_file)
    print("Working with {}".format(file_name))
    data = np.load(data_file)
    
    data_combined = data['event'] + 1e10 * data['run']
    src_combined = np.array(out_dict["gold_event_id"]) + 1e10 * np.array(out_dict["gold_run_id"])
    
    mask = np.in1d(data_combined,src_combined)
    print("Found {} source events".format(np.sum(mask)))
    data_clean = data[~mask]
    outfile = os.path.join(outpath,file_name)
    print("Data file saved to {}".format(outfile))
    np.save(outfile,data_clean)

print("Cleaning GRL datasets")
for data_file in grl_npy_files:
    file_name = os.path.basename(data_file)
    print("Working with {}".format(file_name))
    data = np.load(data_file)
    data_combined = data['events'] + 1e10 * data['run']
    src_combined = np.array(out_dict["gold_event_id"]) + 1e10 * np.array(out_dict["gold_run_id"])

    mask = np.in1d(data_combined,src_combined)
    print("Found {} source events".format(np.sum(mask)))
    data_clean = data[~mask]
    outfile = os.path.join(grl_outpath,file_name)
    print("Data file saved to {}".format(outfile))
    np.save(outfile,data_clean)


print("Cleaning mc data")
inpath = os.path.join(PATHS.data, "check_gold_mc_ids_new")


npy_files = list(glob(os.path.join(inpath,'*.*')))

file_count = len(npy_files)

run_id = []
event_id = []
prim_energy = []
ow = []
azi = []
zen = []

print("Collecting mc ids")
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

# Different versions use different mc over multiple samples
# Read README files for more info

mc_file_name = 'IC86_2016_MC.npy'
mc_files = "../../../../data/ana/analyses/ps_tracks/{}/{}".format(version,mc_file_name)

mc = np.load(mc_files)

test = np.in1d(mc["trueAzi"],out_dict["azi"])
test2 = np.in1d(mc["trueZen"],out_dict["zen"])
test3 = np.in1d(mc["trueE"],out_dict["prim_energy"])
test4 = test*test2*test3
mc_gfu = mc[test4]

mc_clean = mc[~test4]

print('Found {} mc events'.format(np.sum(test4)))

print('That is {}% of mc data'.format(100*np.sum(test4)/len(mc)))

outfile = os.path.join(outpath,mc_file_name)
print("MC file saved to {}".format(outfile))
np.save(outfile,mc_clean)

print("Done")
