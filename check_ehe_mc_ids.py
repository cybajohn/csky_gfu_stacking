# coding:utf-8

"""
Apply EHE filter to sim files and store runIDs, eventIDs and MC primary
energies for all events surviving the filter.
The IDs are checked against the final level MCs to sort out any EHE like events
for sensitivity calulations.
"""

from __future__ import division, print_function

import json
import argparse

from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube import DomTools, weighting
from icecube import VHESelfVeto

# just import everything...
from icecube import icetray, dataclasses, dataio, filterscripts, filter_tools, trigger_sim, WaveCalibrator
from icecube import phys_services, DomTools
from icecube.filterscripts import filter_globals
from icecube.phys_services.which_split import which_split
#from icecube.filterscripts.filter_globals import EHEAlertFilter
from icecube import VHESelfVeto, weighting

from icecube.icetray import I3Units


import sys

#sys.path.append("../")
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from my_tray_scripts import MyEHEFilter_IC86_2012, which_split, ehe_collector, EHEAlertFilter, MyHESEFilter_IC86_2012, hese_collector
from my_tray_scripts import ehe_collector_v2

def main(in_files, out_file_1, out_file_3, out_file_4, gcd_file):
    files = []
    files.append(gcd_file)
    if not isinstance(in_files, list):
        in_files = [in_files]
    files.extend(in_files)

    tray = I3Tray()

    # Read files
    tray.AddModule("I3Reader", "reader", Filenamelist=files)

    #tray.AddSegment(MyHESEFilter_IC86_2012)
    #tray.AddModule(hese_collector, "hese_collector", outfilename = out_file_4)

    #tray.AddSegment(MyPreCalibration_IC86_2012)
    #tray.AddSegment(MyEHECalibration_IC86_2012)
    #tray.AddSegment(MyEHEFilter_IC86_2012, If = which_split(split_name='InIceSplit'))
    #tray.AddModule(CheckFilter, outfilename='filter/result_ehe', filter_key='EHEFilter_12', test_key='MyEHEFilter')
    #tray.AddSegment(EHEAlertFilter,If = which_split(split_name='InIceSplit'))
    #tray.AddModule(weighting.get_weighted_primary, "weighted_primary_again",
    #               If=lambda frame: not frame.Has("MCPrimary"))
    #tray.AddModule(ehe_collector, "ehe_collector",
    #                    outfilename = out_file_1,
    #                    outfilename2 = out_file_2,
    #                    outfilename3 = out_file_3,
    #                    If = which_split(split_name='InIceSplit') 
    #                    )
    tray.AddModule(ehe_collector_v2, "ehe_and_hese_collector",
			outfilename = out_file_1,
			outfilename2 = out_file_3,
			outfilename3 = out_file_4
			)

    tray.AddModule("TrashCan", "NacHsart")
    tray.Execute()
    tray.Finish()



###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Parse options and call `main`
    parser = argparse.ArgumentParser(description="Check EHE and HESE filter")
    parser.add_argument("--infiles", type=str)
    parser.add_argument("--gcdfile", type=str)
    parser.add_argument("--outfile_1", type=str)
    #parser.add_argument("--outfile_2", type=str)
    parser.add_argument("--outfile_3", type=str)
    parser.add_argument("--outfile_4", type=str)

    args = parser.parse_args()

    in_files = args.infiles.split(",")
    gcd_file = args.gcdfile
    out_file_1 = args.outfile_1
    #out_file_2 = args.outfile_2
    out_file_3 = args.outfile_3
    out_file_4 = args.outfile_4
    
    source_type = "ehe" # "ehe" or "hese"
    
    main(in_files, out_file_1, out_file_3, out_file_4, gcd_file)
                                           
