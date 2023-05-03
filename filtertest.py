from __future__ import division, print_function
"""
This script is a test to filter ehe and hese ids from 2012 sim
"""

"""
from __future__ import absolute_import
from icecube import icetray
#from icecube.filterscripts import filter_globals
from I3Tray import *
from icecube import icetray, dataclasses, dataio, filterscripts, filter_tools, trigger_sim, WaveCalibrator
from icecube import phys_services, DomTools
from icecube.filterscripts import filter_globals
from icecube.phys_services.which_split import which_split
#from icecube.filterscripts.filter_globals import EHEAlertFilter
from icecube import VHESelfVeto, weighting

from icecube.icetray import I3Units

import json
import numpy as np

import sys

from my_tray_scripts import MyHESEFilter_IC86_2012, hese_collector

load('VHESelfVeto')

icetray.load("filterscripts",False)

"""
#from __future__ import division, print_function

"""
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
"""

from I3Tray import *
from icecube import icetray
#from icecube.filterscripts import filter_globals
#icetray.load("filterscripts",False)
#icetray.load("cscd-llh",False)

#from icecube import weighting
#from icecube import VHESelfVeto
#from icecube import clast
#from icecube import linefit,lilliput
#from icecube import dataclasses

from icecube import icetray, dataclasses, dataio, filterscripts, filter_tools, trigger_sim, WaveCalibrator
from icecube import payload_parsing
from my_tray_scripts import hese_collector, MyHESEFilter_IC86_2012, MyPreCalibration_IC86_2012, MyEHECalibration_IC86_2012, MyEHEFilter_IC86_2012, EHEAlertFilter

from my_tray_scripts import which_split, CheckFilter

from my_tray_scripts import ehe_collector_v2

filelist = []

#qp_file = "../../../../data/ana/PointSource/PS/version-003-p03/nugen/11070/i3/IC86.2012.011070.002497.upgoing.i3.bz2"
qp_file = "../../../../data/ana/PointSource/PS/version-004-p00/nugen/21220/i3/IC86.2016_NuMu.021220.005372.upgoing.i3.bz2"
gcd_file = "../../../../data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2012.56063_V1.i3.gz"


#files = '../../../data/exp/IceCube/2011/filtered/level2/0731/Level2_IC86.2011_data_Run00118514_Part00000085.i3.bz2'

filelist.append(gcd_file)
filelist.append(qp_file)

outfile = "filter/results_2016"

print("create tray")
tray = I3Tray()

#load('VHESelfVeto')

print("read data")
tray.AddModule("I3Reader", "reader", Filenamelist=filelist)
"""
tray.AddModule("DetectorShrinker",Pulses='InIceDSTPulses',OutPulses='InIceDSTPulsesTrimmed')

# Create correct MCPrimary for energy
tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
                   If=lambda frame: not frame.Has("MCPrimary"))


##########################################################################
# Following code from hesefilter.py
# Prepare Pulses
tray.AddModule("I3LCPulseCleaning", "cleaning_HLC",
                           OutputHLC="InIcePulsesHLC",
                           OutputSLC="",
                           Input="InIceDSTPulsesTrimmed",
                           If=lambda frame: not frame.Has("InIcePulsesHLC"))

# Apply HESE filter
tray.AddModule('VHESelfVeto', 'selfveto', TimeWindow=1500, VertexThreshold=50,
    VetoThreshold=3, Pulses='InIcePulsesHLC', Geometry='I3GeometryTrimmed',If = lambda f: True)
#tray.AddModule("VHESelfVeto",
#                          "selfveto",
#                          Pulses="InIcePulsesHLC",
#                          OutputBool='HESE_VHESelfVeto',
#                          OutputVertexTime='HESE_VHESelfVetoVertexTime',
#                          OutputVertexPos='HESE_VHESelfVetoVertexPos',)

        #tray.AddSegment(HeseFilter, "HESEFilter",
        #                  Pulses="InIcePulses")

# Add CausalQTot frame
tray.AddModule('HomogenizedQTot', 'qtot_causal',
                          Pulses="InIcePulses",
                          Output='HESE_CausalQTot',
                          VertexTime='HESE_VHESelfVetoVertexTime')


#tray.AddSegment(MyHESEFilter_IC86_2012)

#tray.AddModule(hese_collector, "hese_collector", outfilename = outfile)
"""

"""
# Create correct MCPrimary for energy
tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
                           If=lambda frame: not frame.Has("MCPrimary"))


##########################################################################
# Following code from hesefilter.py
# Prepare Pulses
tray.AddModule("I3LCPulseCleaning", "cleaning_HLC",
                           OutputHLC="InIcePulsesHLC",
                           OutputSLC="",
                           Input="InIcePulses",
                           If=lambda frame: not frame.Has("InIcePulsesHLC"))

# Apply HESE filter
tray.AddModule("VHESelfVeto",
                          "selfveto",
                          Pulses="InIcePulsesHLC")

# Add CausalQTot frame
tray.AddModule('HomogenizedQTot', 'qtot_causal',
                          Pulses="InIcePulses",
                          Output='HESE_CausalQTot',
                          VertexTime='VHESelfVetoVertexTime')
"""

"""
tray.AddSegment(payload_parsing.I3DOMLaunchExtractor, 
                          "test" + '_launches',
                          MinBiasID = 'MinBias',
                          FlasherDataID = 'Flasher',
                          CPUDataID = "BeaconHits",
                          SpecialDataID = "SpecialHits",
                          ## location of scintillators and IceACT
                          SpecialDataOMs = [OMKey(0,1),
                                            OMKey(12,65),
                                            OMKey(12,66),
                                            OMKey(62,65),
                                            OMKey(62,66)],
                          )
"""
ehe = True
hese = True

"""
if hese:
	tray.AddSegment(MyHESEFilter_IC86_2012)
	tray.AddModule(hese_collector, "hese_collector", outfilename = outfile)

if ehe:
        #tray.AddSegment(MyPreCalibration_IC86_2012)
        #tray.AddSegment(MyEHECalibration_IC86_2012)
        tray.AddSegment(MyEHEFilter_IC86_2012, If = which_split(split_name='InIceSplit'))
        #tray.AddModule(CheckFilter, outfilename='filter/result_ehe', filter_key='EHEFilter_12', test_key='MyEHEFilter')
        tray.AddSegment(EHEAlertFilter,If = which_split(split_name='InIceSplit'))
        tray.AddModule(CheckFilter, outfilename='filter/result_ehe_alert_HB', filter_key='EHEFilter_12', test_key='EHEAlertFilterHB')
"""
out_file_1 = "filter/filter_result_alert"
out_file_3 = "filter/filter_result_alert_hb"
out_file_4 = "filter/filter_result_hese"



tray.AddModule(ehe_collector_v2, "ehe_and_hese_collector",
                        outfilename = out_file_1,
                        outfilename2 = out_file_3,
                        outfilename3 = out_file_4
                        )


tray.AddModule('I3Writer', 'writer', Filename='filter/filtertest_2_2012.i3.bz2')

#if hese:
#	tray.AddModule(hese_collector, "hese_collector", outfilename = outfile)

tray.AddModule("TrashCan")

tray.Execute()

tray.Finish()

print("Done.")

