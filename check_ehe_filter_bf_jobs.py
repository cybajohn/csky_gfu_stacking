# coding:utf-8

"""
Create jobfiles for `check_ehe_mc_ids.py`.
"""

from __future__ import print_function, division
import warnings
import os
from glob import glob
import numpy as np

from _paths import PATHS
from dagman import dagman


def collect_structure(files):
    """
    Collect data set structure: folder and file paths.
    Assumes: path/[sets]/[folders: eg. 0000-0999]/[files.i3.bz2]
    and puts this strcuture in a dict.

    Parameters
    ----------
    files : dict
        Dictionary with  ``[dataset numbers]`` as keys. Value is a new dict with
        keys, values:

        - 'path', str:  Path to the dataset folders.
          ``path/[folders: eg. 0000-0999]/[files.i3.bz2]``.
        - 'gcd', str: Full path to a GCD file matching the simulation.
        - 'legacy', bool:  If ``True`` all files are assumed to be in a single
          folder structure. This may be useful if the files have been merged and
          moved to an analysis directory and are not available on /data/sim
          anymore. Assumed ``False`` if not given.

    Returns
    -------
    data : dict
        Structure of the dataset in the file system:
        - [sets] - dict
            + "path" - str
            + "folders" - dict
                + [folder] - dict
                    + "files" - list
                    + "nfiles" (in folder) - int
            + "nfiles" (in dataset) - int
            + "gcd" - str
    """
    data = {}
    for num in sorted(list(files.keys())):
        path = os.path.abspath(files[num]["path"])
        GCD = os.path.abspath(files[num]["gcd"])
        LEGACY = files[num].get("legacy", False)

        if not os.path.isfile(GCD):
            raise RuntimeError("GCD file '{}' doesn't exist.".format(GCD))
        if not LEGACY:  # Assume subfolders 00000-09999, ...
            folders = [os.path.join(path, s) for s in sorted(os.listdir(path))]
            folders = filter(os.path.isdir, folders)
        else:  # Assume everything is in this folder
            folders = [os.path.join(path, ".")]
        data[num] = {"path": path}
        data[num]["gcd"] = GCD
        data[num]["folders"] = {}

        print("\nDataset: ", num)
        print(" - Path:\n   ", path)
        print(" - GCD:\n   ", GCD)
        print(" - Folders:")
        nfiles = 0
        for folder in folders:
            folder_name = os.path.basename(folder)
            # Don't collect GCDs here but manually below
            sim_files = sorted([f for f in glob(folder + "/*.i3.zst") if
                                not f.startswith("Geo")])
            data[num]["folders"][folder_name] = {"files": sim_files,
                                                 "nfiles": len(sim_files)}
            print("   + ", folder_name, ": ", len(sim_files), " files")
            nfiles += len(sim_files)
        data[num]["nfiles"] = nfiles
        print(" - Total files: ", nfiles)
        if nfiles == 0:
            raise RuntimeError("No files found for set '{}'. Make sure " +
                               "locations are valid or remove set from list.")

    return data


if __name__ == "__main__":
    # Job creation steering arguments
    job_creator = dagman.DAGManJobCreator(mem=3) # one job required 2400MB
    
    #source_type = "ehe" # adjust for 'ehe' or 'hese'
     
    job_name = "ehe_transient_stacking"
    
    job_dir = os.path.join(PATHS.jobs, "check_ehe_mc_ids_bf")
    script = os.path.join(PATHS.repo, "check_ehe_mc_ids.py")

    ###########################################################################
    # Collect dataset structure. Datset info on:
    #   https://grid.icecube.wisc.edu/simulation/dataset/<number>
    ###########################################################################
    print("\nCollecting simulation file paths:")
    # MC used for:
    # Point Source tracks IC86 2012-2014 datasets
    # GFU tracks IC86 2015-2017 datasets
    # Available as i3 on final level
    fpath = os.path.join("/data", "ana", "realtime", "alert_catalog_v2")
    gcd_path = os.path.join("/data", "sim", "sim-new", "downloads", "GCD")
    
    # take files blaufuss told me, gcd is irrelevant but Ill keep em here
    files = {
        "20878": {
            "path": os.path.join(fpath, "sim_20878_alerts"),
            "gcd": os.path.join(gcd_path, "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": True,
            },
        "21002": {
            "path": os.path.join(fpath, "sim_21002_alerts"),
            "gcd": os.path.join(gcd_path, "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": True,
            },
        "21220": {
            "path": os.path.join(fpath, "sim_21220_alerts"),
            "gcd": os.path.join(gcd_path, "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": True,
            }
        }
    # Combine to a single dict listing all files and folders
    data = collect_structure(files)

    ###########################################################################
    # Make job argument lists
    ###########################################################################
    # Job splitting args and job output paths
    nfiles_perjob = 100
    # make 3 outpath for checking purposes
    outpath_1 = os.path.join(PATHS.data, "check_ehe_alert_mc_ids_bf")
    #outpath_2 = os.path.join(PATHS.data, "check_my_ehe_mc_ids")
    outpath_3 = os.path.join(PATHS.data, "check_ehe_alert_hb_mc_ids_bf")
    outpath_4 = os.path.join(PATHS.data, "check_hese_mc_ids_bf")
    if os.path.isdir(outpath_1) and os.path.isdir(outpath_3) and os.path.isdir(outpath_4):
        print("")
        warnings.warn("Output folder '{}' is already ".format(outpath_1) +
                      "existing. Check twice if nothing gets overwritten " +
                      "when starting jobs.", RuntimeWarning)
    else:
        os.makedirs(outpath_1)
	#os.makedirs(outpath_2)
	os.makedirs(outpath_3)
	os.makedirs(outpath_4)

    # Prepare job files by splitting arg list to jobfiles
    file_list = []
    gcd_list = []
    out_list_1 = []
    #out_list_2 = []
    out_list_3 = []
    out_list_4 = []
    print("")
    for num in data.keys():
        # Combine all files for the current sample
        folders = data[num]["folders"]
        files = np.concatenate([folder["files"] for folder in folders.values()])
        nfiles = len(files)
        assert nfiles == data[num]["nfiles"]

        # Split in chunks of ca. 10 files per job so jobs run fast
        nsplits = nfiles // nfiles_perjob
        chunked = np.array_split(files, nsplits)
        file_list.append(map(",".join, chunked))
        njobs = len(chunked)
        assert njobs == len(file_list[-1])
        print("Set: ", num)
        _min, _max = np.amin(map(len, chunked)), np.amax(map(len, chunked))
        if _min == _max:
            print("  Split {} files to {} jobs with {} ".format(
                nfiles, njobs, _min) + "files per job.")
        else:
            print("  Split {} files to {} jobs with {} - {} ".format(
                nfiles, njobs, _min, _max) + "files per job.")

        # Duplicate GCD info per job
        gcd_list.append(njobs * [data[num]["gcd"]])
        assert njobs == len(gcd_list[-1])
        assert np.all(np.array(gcd_list[-1]) == gcd_list[-1][0])

        # Outpath: ..[num]_<increment>.json
        lead_zeros = int(np.ceil(np.log10(nsplits)))
        outp = ["{2:}_{1:0{0:d}d}.json".format(lead_zeros, idx, num) for
                idx in np.arange(nsplits)]
	# not elegant but well...
        out_list_1.append([os.path.join(outpath_1, pi) for pi in outp])
	#out_list_2.append([os.path.join(outpath_2, pi) for pi in outp])
	out_list_3.append([os.path.join(outpath_3, pi) for pi in outp])
	out_list_4.append([os.path.join(outpath_4, pi) for pi in outp])

    # Compress and write job files
    in_files = []
    map(in_files.extend, file_list)
    gcd_files = []
    map(gcd_files.extend, gcd_list)
    out_paths_1 = []
    map(out_paths_1.extend, out_list_1)
    #out_paths_2 = []
    #map(out_paths_2.extend, out_list_2)
    out_paths_3 = []
    map(out_paths_3.extend, out_list_3)
    out_paths_4 = []
    map(out_paths_4.extend, out_list_4)


    assert len(in_files) == len(out_paths_1) == len(gcd_files)
    print("\nTotal number of jobs: ", len(out_paths_1), "\n")

    job_args = {"infiles": in_files, "gcdfile": gcd_files, "outfile_1": out_paths_1, "outfile_3": out_paths_3, "outfile_4": out_paths_4}

    exe = [
        os.path.join("/home", "jkollek", "software", "build",
                     "env-shell.sh"),
        os.path.join("/bin", "bash")
    ]
    job_creator.create_job(script=script, job_args=job_args, exe=exe,
                           job_name=job_name, job_dir=job_dir, overwrite=True) # set False later
