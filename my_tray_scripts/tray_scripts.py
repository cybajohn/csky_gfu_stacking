from icecube import icetray, dataclasses
from icecube import VHESelfVeto, weighting
#from icecube.filterscripts.hesefilter import HeseFilter
import json
import math

def which_split(split_name = None,split_names=[]):
        """
        Function to select sub_event_streams on which modules shall run on
        
        Parameters
        ----------
        split_name: str (optional)
                name of sub_event_stream
        split_names: list (optional)
                list of names of sub_event_streams
        
        Return
        ------
        which_split: bool
                True if stream is of chosen type of sub_event_streams, else False
        """
        if split_name:
                split_names.append(split_name)
        def which_split(frame):
                if len(split_names)==0:
                        print "Select a split name in your If...Running module anyway"
                        return True
                if frame.Stop == icetray.I3Frame.Physics:
                        return (frame['I3EventHeader'].sub_event_stream in split_names)
                else:
                        return False
        return which_split

class CheckFilter(icetray.I3ConditionalModule):
        """
        Checks and counts if filters were passed like previous ones in 'QFilterMask'
        and writes the results as a dict into a json file
        
        Parameters
        ----------
        outfilename: str,
                name of outputfile
        filter_key: str,
                name of key in 'QFilterMask'
        test_key: str,
                name of test key
        
        Return
        ------
        json file containing resulting dict
        """
        def __init__(self,context):
                icetray.I3ConditionalModule.__init__(self, context)
                self.AddParameter("outfilename","outfilename","")
                self.AddParameter("filter_key","filter_key","")
                self.AddParameter("test_key","test_key","")
        def Configure(self):
                self.outfile = self.GetParameter("outfilename")
                self.filter_key = self.GetParameter("filter_key")
                self.test_key = self.GetParameter("test_key")
                self.filter_key_count, self.test_key_count, self.both_count = 0, 0, 0
        def Physics(self, frame):
                if frame.Has(self.test_key):
                        if frame[self.test_key].value:
                                self.test_key_count += 1
                                if frame["QFilterMask"][self.filter_key].condition_passed:
                                        self.both_count += 1
                        if frame["QFilterMask"][self.filter_key].condition_passed:
                                self.filter_key_count += 1
                        self.PushFrame(frame)
        def Finish(self):
                out_dict = {self.test_key:self.test_key_count, self.filter_key:self.filter_key_count, "both":self.both_count}
                with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
                print("Wrote output file to:\n ", self.outfile)


@icetray.traysegment
def MyHESEFilter_IC86_2012(tray, name="", If=lambda f:True):
        from icecube.filterscripts import filter_globals
        icetray.load("filterscripts",False)
        icetray.load("cscd-llh",False)
        
        from icecube import weighting
        from icecube import VHESelfVeto
        from icecube import clast
        from icecube import linefit,lilliput #unnecessary but Ill leave it
        from icecube import dataclasses
        
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
                          Pulses="InIcePulsesHLC",
                          OutputBool='VHESelfVeto',
                          OutputVertexTime='HESE_VHESelfVetoVertexTime',
                          OutputVertexPos='HESE_VHESelfVetoVertexPos',)

	#tray.AddSegment(HeseFilter, "HESEFilter",
        #                  Pulses="InIcePulses")
        
        # Add CausalQTot frame
        tray.AddModule('HomogenizedQTot', 'qtot_causal',
                          Pulses="InIcePulses",
                          Output='HESE_CausalQTot',
                          VertexTime='HESE_VHESelfVetoVertexTime')



class hese_collector(icetray.I3ConditionalModule):
    """
    Collect run ID, event ID and MC primary energy for events surviving the
    HESE filter.
    """

    def __init__(self, ctx):
        icetray.I3ConditionalModule.__init__(self, ctx)
        self.AddParameter("outfilename", "outfilename", "")

    def Configure(self):
        # HESE total charge cut: wiki.icecube.wisc.edu/index.php/HESE-7year#Summary_of_Cuts_and_Parameters
        self.minimum_charge = 6000.

        self.outfile = self.GetParameter("outfilename")
        self.run_id = []
        self.event_id = []
        # Just save some extra stuff
        self.qtot = []
        self.energy = []

    def Physics(self, frame):
        # If HESE veto is passed, VHESelfVeto variable is False.
        # Also VHESelfVeto doesn't always write the key, which means the event
        # is vetoed.
        try:
            hese_veto = frame["VHESelfVeto"].value
        except KeyError:
            hese_veto = True
        if not hese_veto:
            # Not HESE vetoed: Candidate event
            qtot = frame["HESE_CausalQTot"].value
            if qtot >= self.minimum_charge:
                # Also over QTot cut: Winner
                evt_header = frame["I3EventHeader"]
                prim = frame["MCPrimary"]

                self.run_id.append(evt_header.run_id)
                self.event_id.append(evt_header.event_id)
                self.energy.append(prim.energy)
                self.qtot.append(qtot)

        self.PushFrame(frame)

    def Finish(self):
        out_dict = {"energy": self.energy, "run_id": self.run_id,
                    "event_id": self.event_id, "qtot": self.qtot}
        with open(self.outfile, "w") as outf:
            json.dump(out_dict, fp=outf, indent=2)
        print("Wrote output file to:\n  ", self.outfile)

class ehe_collector_v2(icetray.I3ConditionalModule):
        """
        Testversion of ehe_collector, writes out keys of ehe_alert, ehe_alert_hb and hese alerts

        Parameters
        ----------
        outfilename: str
                name of savefile for ehe_alert
        outfilename2: str
                name of savefile for ehe_alert_hb
        outfilename3: str
                name of savefile for hese_alert
        
        Return
        ------
        savefiles for the event types with energy, run_id, event_id
        """
        def __init__(self, ctx):
                icetray.I3ConditionalModule.__init__(self, ctx)
                self.AddParameter("outfilename","outfilename","")
                self.AddParameter("outfilename2","outfilename2","")
                self.AddParameter("outfilename3","outfilename3","")

        def Configure(self):
                self.outfile = self.GetParameter("outfilename")
                self.second_outfile = self.GetParameter("outfilename2")
                self.third_outfile = self.GetParameter("outfilename3")
                # alert_filter
		self.alert_run_id = []
                self.alert_event_id = []

                # alert_filter_bh
                self.alert_hb_run_id = []
                self.alert_hb_event_id = []

                # hese
                self.hese_run_id = []
                self.hese_event_id = []

        def Physics(self, frame):
                try:
                        ehe_alert = frame["QFilterMask"]["EHEAlertFilter_15"].condition_passed
                except KeyError:
                        ehe_alert = False
                try:
                        ehe_alert_hb = frame["QFilterMask"]["EHEAlertFilterHB_15"].condition_passed
                except KeyError:
                        ehe_alert_hb = False
                try:
                        hese_alert = frame["QFilterMask"]["HESEFilter_15"].condition_passed
                except KeyError:
                        hese_alert = False

                if ehe_alert:
                        evt_header = frame["I3EventHeader"]
                        self.alert_run_id.append(evt_header.run_id)
                        self.alert_event_id.append(evt_header.event_id)
                if ehe_alert_hb:
                        evt_header = frame["I3EventHeader"]
                        self.alert_hb_run_id.append(evt_header.run_id)
                        self.alert_hb_event_id.append(evt_header.event_id)
                if hese_alert:
                        evt_header = frame["I3EventHeader"]
                        self.hese_run_id.append(evt_header.run_id)
                        self.hese_event_id.append(evt_header.event_id)
                self.PushFrame(frame)

        def Finish(self):
                out_dict = {"run_id": self.alert_run_id, "event_id": self.alert_event_id}
                out_dict2 = {"run_id": self.alert_hb_run_id, "event_id": self.alert_hb_event_id}
                out_dict3 = {"run_id": self.hese_run_id, "event_id": self.hese_event_id}
                with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
                with open(self.second_outfile, "w") as outf:
                        json.dump(out_dict2, fp=outf, indent=2)
                with open(self.third_outfile, "w") as outf:
                        json.dump(out_dict3, fp=outf, indent=2)
                print("Wrote output file to:\n ", self.outfile, self.second_outfile, self.third_outfile)




class ehe_collector(icetray.I3ConditionalModule):
        """
        Testversion of ehe_collector, writes out keys of ehe, my_ehe and ehe alerts for comparision

        Parameters
        ----------
        outfilename: str
                name of savefile for ehe
        outfilename2: str
                name of savefile for my ehe
        outfilename3: str
                name of savefile for my ehe alert
        
        Return
        ------
        savefiles for the event types with energy, run_id, event_id
        """
        def __init__(self, ctx):
                icetray.I3ConditionalModule.__init__(self, ctx)
                self.AddParameter("outfilename","outfilename","")
                self.AddParameter("outfilename2","outfilename2","")
                self.AddParameter("outfilename3","outfilename3","")

        def Configure(self):
                self.outfile = self.GetParameter("outfilename")
                self.second_outfile = self.GetParameter("outfilename2")
                self.alert_outfile = self.GetParameter("outfilename3")
                self.run_id = []
                self.event_id = []
                # extra stuff Thorben saved
                self.energy = []        # only accessible through the prior use of weighting ["MCPrimary"]

                # for sanity checks
                self.my_run_id = []
                self.my_event_id = []
                self.my_energy = []

                # alert_filter
                self.alert_run_id = []
                self.alert_event_id = []
                self.alert_energy = []

        def Physics(self, frame):
                try:
                        ehe_like = frame["QFilterMask"]["EHEFilter_12"].condition_passed
                except KeyError:
                        try:
                                ehe_like = frame["FilterMask"]["EHEFilter_11"].condition_passed
                        except KeyError:
                                try:
                                        ehe_like = frame["FilterMask"]["EHEFilter_10"].condition_passed
                                except KeyError:
                                        print("Unable to find ehe_filter, setting it to true")
                                        ehe_like = True
                try:
                        own_ehe_like = frame["MyEHEFilter"].value
                except KeyError:
                        own_ehe_like = 0

                try:
                        ehe_alert = frame["EHEAlertFilterHB"].value
                except KeyError:
                        ehe_alert = 2 # for distinguishing reasons uwu

                if ehe_like:
                        evt_header = frame["I3EventHeader"]
                        prim = frame["MCPrimary"]

                        self.run_id.append(evt_header.run_id)
                        self.event_id.append(evt_header.event_id)

                        self.energy.append(prim.energy)

                        #self.PushFrame(frame)

                if own_ehe_like:
                        evt_header = frame["I3EventHeader"]
                        prim = frame["MCPrimary"]
                        self.my_run_id.append(evt_header.run_id)
                        self.my_event_id.append(evt_header.event_id)
                        self.my_energy.append(prim.energy)

                if ehe_alert == 1:
                        evt_header = frame["I3EventHeader"]
                        prim = frame["MCPrimary"]
                        self.alert_run_id.append(evt_header.run_id)
                        self.alert_event_id.append(evt_header.event_id)
                        self.alert_energy.append(prim.energy)

                self.PushFrame(frame)

        def Finish(self):
                out_dict = {"energy": self.energy, "run_id": self.run_id, "event_id": self.event_id}
                out_dict2 = {"energy": self.my_energy, "run_id": self.my_run_id, "event_id": self.my_event_id}
                out_dict3 = {"energy": self.alert_energy, "run_id": self.alert_run_id, "event_id": self.alert_event_id}
                with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
                with open(self.second_outfile, "w") as outf:
                        json.dump(out_dict2, fp=outf, indent=2)
                with open(self.alert_outfile, "w") as outf:
                        json.dump(out_dict3, fp=outf, indent=2)
                print("Wrote output file to:\n ", self.outfile, self.second_outfile, self.alert_outfile)


@icetray.traysegment
def MyPreCalibration_IC86_2012(tray, name="", If=lambda f:True):
        """
        Segment I cobbled together to generate necessary keys for the ehe calibration,
        works for IC86_2011 to GFU 2015 sample.
        Somehow IC79 still refuses to bow to my will...
        """
        from icecube import icetray, dataclasses
        from icecube.filterscripts import filter_globals

        def rename(frame):
                """ Rename for older files not having 'InIcePulses' key """
                if not frame.Has("InIcePulses"): # not python-ish, but understandable
                        frame["InIcePulses"] = frame["OfflinePulses"]
                return True

        def CheckIfNoDSTTriggers(frame):
                """
                Checks if frame has 'DSTTriggers'
                """
                if frame.Has("DSTTriggers"):
                        return False
                return True


        def TriggerPacker(frame):
                """
                Create a compressed representation of the trigger hierarchy,
                using the position of the TriggerKey in
                I3DetectorStatus::triggerStatus to identify each trigger.
                'I3TriggerHierarchy' is old these days
                """
                triggers = frame["I3TriggerHierarchy"]
                status = frame['I3DetectorStatus']
                packed = dataclasses.I3SuperDSTTriggerSeries(triggers, status)
                frame['DSTTriggers'] = packed

        DAQ = [icetray.I3Frame.DAQ]

        #tray.AddModule(rename_q_frame_key, "Rename_Trigger_Hier",
        #                old_key = "I3TriggerHierarchy",
        #                new_key = "DSTTriggers",
        #                If = CheckIfNoDSTTriggers
        #                )

        tray.AddModule(rename, "rename_offline_pulses"
                        )

        # old files name 'InIcePulses' 'OfflinePulses', rename them in the q-frame
        #tray.AddModule(rename_q_frame_key, "rename_pulses",
        #                old_key = "OfflinePulses",
        #                new_key = "InIcePulses"
        #                )

        # I3 portia only works on 'InIceSplit' + we need the time window, which is
        # never really mentioned where to get it from, but the last line will do the trick
        tray.AddModule("I3TriggerSplitter", "InIceSplit",
                        SubEventStreamName = "InIceSplit",
                        TrigHierName = "DSTTriggers",
                        #InputResponses = ["InIcePulses"],
                        #OutputResponses = ["SplitInIcePulses"],
                        #InputResponses = ["InIceDSTPulses", "InIcePulses"],
                        #OutputResponses = ["SplitInIceDSTPulses", "SplitInIcePulses"],
                        WriteTimeWindow = True,
                        )

        # Lets try this here:
        # TriggerCheck important for EHEFlag, setting the things we need
        tray.AddModule("TriggerCheck_13", "TriggerChecker",
                        I3TriggerHierarchy = "DSTTriggers",
                        InIceSMTFlag = "InIceSMTTriggered")


        # CleanInIceRawData
        icetray.load("DomTools", False)
        tray.AddModule("I3DOMLaunchCleaning", "I3LCCleaning",
                        InIceInput = "InIceRawData",
                        InIceOutput = "CleanInIceRawData",
                        IceTopInput = "IceTopRawData",
                        IceTopOutput = "CleanIceTopRawData",
#                       If = which_split(split_names = ["InIceSplit", "nullsplit"])
                        )

        # seeding which might be unnecessary here
        from icecube.icetray import I3Units
        from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
        seededRTConfig = I3DOMLinkSeededRTConfigurationService(
                        ic_ic_RTRadius              = 150.0*I3Units.m,
                        ic_ic_RTTime                = 1000.0*I3Units.ns,
                        treat_string_36_as_deepcore = False,
                        useDustlayerCorrection      = False,
                        allowSelfCoincidence        = True
                        )

        tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'North_seededrt',
                        InputHitSeriesMapName  = 'SplitInIcePulses',
                        OutputHitSeriesMapName = 'SRTInIcePulses',
                        STConfigService        = seededRTConfig,
                        SeedProcedure          = 'HLCCoreHits',
                        NHitsThreshold         = 2,
                        MaxNIterations         = 3,
                        Streams                = [icetray.I3Frame.Physics],
                        If = If
                        )


def MyEHECalibration_IC86_2012(tray, name="", inPulses = 'CleanInIceRawData',
                   outATWD = 'EHECalibratedATWD_Wave', outFADC = 'EHECalibratedFADC_Wave',
                   PreIf = lambda f: True,
                   If = lambda f: True,):
        from icecube import icetray, dataclasses, DomTools, ophelia, WaveCalibrator
        from icecube.icetray import OMKey

        tray.AddSegment(MyPreCalibration_IC86_2012, "GeneratingKeysForCalibration",
                        If = PreIf)
        #**************************************************************
        ### Run WaveCalibrator w/o droop correction. DeepCore DOMs are omitted.
        ### Split waveforms up into two maps FADC and ATWD (highest-gain unsaturated channel)
        #***
        #***********************************************************
        # temporal. should not be needed in the actual script
        #tray.AddModule("I3EHEEventSelector", name + "inicePframe",setCriteriaOnEventHeader = True, If=If)
        # This may not be needed if the frame have HLCOfflineCleanInIceRawData
        # Actually, we like to have the bad dom cleaned launches here
        tray.AddModule( "I3LCCleaning", name + "OfflineInIceLCCleaningSLC",
                        InIceInput = inPulses,
                        InIceOutput = "HLCOfflineCleanInIceRawData",  # ! Name of HLC-only DOMLaunches
                        InIceOutputSLC = "SLCOfflineCleanInIceRawData",  # ! Name of the SLC-only DOMLaunches
                        If = If,)
        #**************************************************************
        # removing Deep Core strings
        #**************************************************************
        tray.AddModule("I3DOMLaunchCleaning", name + "LaunchCleaning",
                        InIceInput = "HLCOfflineCleanInIceRawData",
                        InIceOutput = "HLCOfflineCleanInIceRawDataWODC",
                        CleanedKeys = [OMKey(a,b) for a in range(79, 87) for b in range(1, 61)],
                        IceTopInput = "CleanIceTopRawData", #nk: Very important! otherwise it re-cleans IT!!!
                        IceTopOutput = "CleanIceTopRawData_EHE", #nk: okay so this STILL tries to write out IceTop.. give different name
                        If = If,)

        #***********************************************************
        # Calibrate waveforms without droop correction
        #***********************************************************
        tray.AddModule("I3WaveCalibrator", name + "calibrator",
                        Launches="HLCOfflineCleanInIceRawDataWODC",
                        Waveforms="EHEHLCCalibratedWaveforms",
                        ATWDSaturationMargin=123, # 1023-900 == 123
                        FADCSaturationMargin=0,
                        CorrectDroop=False,
                        WaveformRange="", #nk: don't write out Calibrated Waveform Range... already written with default name by Recalibration.py
                        If = If, )
                                        
        tray.AddModule("I3WaveformSplitter", name + "split",
                        Input="EHEHLCCalibratedWaveforms",
                        HLC_ATWD = outATWD,
                        HLC_FADC = outFADC,
                        SLC = "EHECalibratedSLC",
                        Force=True,
                        PickUnsaturatedATWD=True,
                        If = If, )

def SelectOMKeySeedPulse(omkey_name, pulses_name, seed_name):
        def do(f):
                om = f[omkey_name][0]
                f[seed_name] = dataclasses.I3RecoPulseSeriesMapMask(f, pulses_name, lambda omkey, p_idx, p: omkey == om and p_idx == 0)
        return do

@icetray.traysegment
def MyEHEFilter_IC86_2012(tray, name="",
                        inATWD = 'EHECalibratedATWD_Wave', inFADC = 'EHECalibratedFADC_Wave',
                        PreIf = lambda f: True,
                        CalibIf = lambda f: True,
                        If = lambda f: True):
        tray.AddSegment(MyEHECalibration_IC86_2012, "EHECalibration",
                        PreIf = PreIf,
                        If = CalibIf
                        )
        from icecube.icetray import I3Units
        from icecube import STTools
        #icetray.load("libSeededRTCleaning");
        # Create a SeededRT configuration object with the standard RT settings.
        # This object will be used by all the different SeededRT modules of this EHE
        # hit cleaning tray segment.
        from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
        seededRTConfigEHE = I3DOMLinkSeededRTConfigurationService(
                ic_ic_RTRadius              = 150.0*I3Units.m,
                ic_ic_RTTime                = 1000.0*I3Units.ns,
                treat_string_36_as_deepcore = False,
                useDustlayerCorrection      = True, # EHE use the dustlayer correction!
                allowSelfCoincidence        = True
        )
        #***********************************************************
        # portia splitter
        # This module takes the splitted start time and end time and makes split DOM map
        #***********************************************************
        tray.AddModule("I3PortiaSplitter", name + "EHE-SplitMap-Maker",
                        DataReadOutName="HLCOfflineCleanInIceRawDataWODC",
                        SplitDOMMapName="splittedDOMMap",
                        SplitLaunchTime=True,
                        TimeWindowName = "TriggerSplitterLaunchWindow",
                        If = If
                        )
        #***************************************************************                                   
        #     Portia Pulse process  with the split DOM map for SeededRT seed
        #***************************************************************                                   
        tray.AddModule("I3Portia", name + "pulseSplitted",
                        SplitDOMMapName = "splittedDOMMap",
                        OutPortiaEventName = "EHEPortiaEventSummary",
                        ReadExternalDOMMap=True,
                        MakeIceTopPulse=False,
                        ATWDPulseSeriesName = "EHEATWDPulseSeries",
                        ATWDPortiaPulseName = "EHEATWDPortiaPulse",
                        ATWDWaveformName = inATWD,
                        ATWDBaseLineOption = "eheoptimized",
                        FADCBaseLineOption = "eheoptimized",
                        ATWDThresholdCharge = 0.1*I3Units.pC,
                        ATWDLEThresholdAmplitude = 1.0*I3Units.mV,
                        UseFADC = True,
                        FADCPulseSeriesName = "EHEFADCPulseSeries",
                        FADCPortiaPulseName = "EHEFADCPortiaPulse",
                        FADCWaveformName = inFADC,
                        FADCThresholdCharge = 0.1*I3Units.pC,
                        FADCLEThresholdAmplitude = 1.0*I3Units.mV,
                        MakeBestPulseSeries = True,
                        BestPortiaPulseName = "EHEBestPortiaPulse",
                        PMTGain = 10000000,
                        If = which_split(split_name='InIceSplit')
                        )

        tray.AddModule("I3PortiaEventOMKeyConverter", name + "portiaOMconverter",
                        InputPortiaEventName = "EHEPortiaEventSummary",
                        OutputOMKeyListName = "LargestOMKey",
                        If = If
                        )

        #***************************************************************
        #     EHE SeededRTCleaning for DOMmap
        #***************************************************************
        #---------------------------------------------------------------------------
        # The splittedDOMMap frame object is an I3MapKeyVectorDouble
        # object. In order to use STTools in the way SeededRTCleaning did it, we
        # need to convert this into an I3RecoPulseSeriesMap where the time of each
        # pulse is set to the double value of the splittedDOMMap frame object.
        def I3MapKeyVectorDouble_to_I3RecoPulseSeriesMap(f):
                i3MapKeyVectorDouble = f['splittedDOMMap']
                i3RecoPulseSeriesMap = dataclasses.I3RecoPulseSeriesMap()
                for (k,l) in i3MapKeyVectorDouble.items():
                        pulses = dataclasses.I3RecoPulseSeries()
                        for d in l:
                                p = dataclasses.I3RecoPulse()
                                p.time = d
                                pulses.append(p)
                        i3RecoPulseSeriesMap[k] = pulses
                f['splittedDOMMap_pulses'] = i3RecoPulseSeriesMap
        # Lets try this, maybe it works (hopefully)
        # make EHE decision based on 10**3 npe
        tray.AddModule("I3FilterModule<I3EHEFilter_13>","EHEfilter",
                        TriggerEvalList=["InIceSMTTriggered"],
                        DecisionName    = "MyEHEFilter",
                        DiscardEvents   = False,
                        PortiaEventName = "EHEPortiaEventSummary",
                        Threshold       = pow(10,3.0),
                        If = which_split(split_name='InIceSplit')
                        )

# function and segment below are copied from 
# https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/combo/trunk/filterscripts/python/ehealertfilter.py
#  EHEAlertFilter -- Only run if NPE threshold high enough                             
# Note: this threshold higher than EHEFilter   
def RunEHEAlertFilter(frame):
        if 'EHEPortiaEventSummary' not in frame:
                return False
        # Check EHE filter first, it's low rate to start...bool in frame here         
        try:
                ehefilterflag = frame["QFilterMask"]["EHEFilter_12"].condition_passed
        except KeyError:
                try:
                        ehefilterflag = frame["FilterMask"]["EHEFilter_11"].condition_passed
                except KeyError:
                        try:
                                ehefilterflag = frame["FilterMask"]["EHEFilter_10"].condition_passed
                        except KeyError:
                                return False
        if not ehefilterflag:
                return False
        npe    = frame['EHEPortiaEventSummary'].GetTotalBestNPEbtw()
        if math.isnan(npe): return False
        if npe <= 0:        return False
        lognpe = math.log10(npe)
        return lognpe >= 3.6

@icetray.traysegment
def EHEAlertFilter(tray, name='',
                   pulses         = 'CleanedMuonPulses',
                   portia_pulse   = 'EHEBestPortiaPulse',   # Maybe this should be 'Pole'
                   portia_summary = 'EHEPortiaEventSummary',
                   split_dom_map  = 'splittedDOMMap',
                   If = lambda f: True):
        
        # Some necessary stuff
        from icecube import dataclasses, linefit
        from icecube.icetray import OMKey, I3Units
        from icecube.filterscripts import filter_globals
        icetray.load("filterscripts",False)
        icetray.load("portia",False)
        icetray.load("ophelia",False)
        from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
        """
        # Get the largest OM Key in the frame
        tray.AddModule("I3PortiaEventOMKeyConverter", name + "portiaOMconverter",
                       InputPortiaEventName = portia_summary,
                       OutputOMKeyListName = "LargestOMKey",
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        """
        # Static time window cleaning
        tray.AddModule("I3EHEStaticTWC", name + "portia_static",
                       InputPulseName = portia_pulse,
                       InputPortiaEventName = portia_summary,
                       outputPulseName = name + "BestPortiaPulse_BTW",
                       TimeInterval = 500.0 * I3Units.ns, #650, now no interval cut
                       TimeWidnowNegative =  -2000.0 * I3Units.ns,
                       TimeWindowPositive = 6400.0 * I3Units.ns,
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Convert portia pulses
        tray.AddModule("I3OpheliaConvertPortia", name + "portia2reco",
                       InputPortiaPulseName = name + "BestPortiaPulse_BTW",
                       OutputRecoPulseName = name + "RecoPulseBestPortiaPulse_BTW",
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Run delay cleaning
        tray.AddModule("DelayCleaningEHE", name + "DelayCleaning",
                       InputResponse = name + "RecoPulseBestPortiaPulse_BTW",
                       OutputResponse = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay",
                       Distance = 200.0 * I3Units.m, #156m default
                       TimeInterval = 1800.0 * I3Units.ns, #interval 1.8msec
                       TimeWindow = 778.0 * I3Units.ns, #778ns default
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        seededRTConfigEHE = I3DOMLinkSeededRTConfigurationService(
            ic_ic_RTRadius              = 150.0*I3Units.m,
            ic_ic_RTTime                = 1000.0*I3Units.ns,
            treat_string_36_as_deepcore = False,
            useDustlayerCorrection      = True, # EHE use the dustlayer correction!
            allowSelfCoincidence        = True
            )

        tray.AddModule('I3SeededRTCleaning_RecoPulse_Module', name+'Isolated_DelayClean',
                       InputHitSeriesMapName  = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay",
                       OutputHitSeriesMapName = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
                       STConfigService        = seededRTConfigEHE,
                       SeedProcedure          = 'HLCCOGSTHits',
                       MaxNIterations         = -1, # Infinite.
                       Streams                = [icetray.I3Frame.Physics],
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Huber fit
        tray.AddModule("HuberFitEHE", name + "HuberFit",
                       Name = "HuberFit",
                       Distance = 180.0 * I3Units.m, #153m default
                       InputRecoPulses = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Debiasing of pulses
        tray.AddModule("DebiasingEHE", name + "Debiasing",
                       OutputResponse = name + "debiased_BestPortiaPulse_CleanDelay",
                       InputResponse = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
                       Seed = "HuberFit",
                       Distance = 150.0 * I3Units.m,#116m default
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Convert back to portia pulses to be fed to ophelia
        tray.AddModule("I3OpheliaConvertRecoPulseToPortia", name + "reco2portia",
                       InputRecoPulseName = name + "debiased_BestPortiaPulse_CleanDelay",
                       OutputPortiaPulseName = name + "portia_debiased_BestPortiaPulse_CleanDelay",
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Run I3EHE First guess module and get final result
        tray.AddModule("I3EHEFirstGuess", name + "reco_improvedLinefit",
                       MinimumNumberPulseDom = 8,
                       InputSplitDOMMapName = split_dom_map,
                       OutputFirstguessName = "PoleEHEOphelia_ImpLF", # Final result
                       OutputFirstguessNameBtw = name + "OpheliaBTW_ImpLF", # Don't Use
                       InputPulseName1 = name + "portia_debiased_BestPortiaPulse_CleanDelay",
                       ChargeOption = 1,
                       LCOption =  False,
                       InputPortiaEventName =portia_summary,
                       OutputParticleName = "PoleEHEOpheliaParticle_ImpLF", # Final Result
                       OutputParticleNameBtw = name + "OpheliaParticleBTW_ImpLF", # Don't Use
                       NPEThreshold = 0.0,
                       If = (If and (lambda f:RunEHEAlertFilter(f)))
                       )

        # Run the alert filter
        tray.AddModule("I3FilterModule<I3EHEAlertFilter_15>","EHEAlertFilter",
                       TriggerEvalList= ["InIceSMTTriggered"],
                       DecisionName = "EHEAlertFilter",
                       DiscardEvents = False,
                       PortiaEventName = portia_summary,
                       EHEFirstGuessParticleName = "PoleEHEOpheliaParticle_ImpLF",
                       EHEFirstGuessName = "PoleEHEOphelia_ImpLF",
                       If = (If and (lambda f: 'PoleEHEOphelia_ImpLF' in f) # First Guess can fail
                             and (lambda f:RunEHEAlertFilter(f)))
                       )
        # Again for Heartbeat
        tray.AddModule("I3FilterModule<I3EHEAlertFilter_15>","EHEAlertFilterHB",
                       TriggerEvalList = ["InIceSMTTriggered"],
                       DecisionName = "EHEAlertFilterHB",
                       DiscardEvents = False,
                       PortiaEventName = portia_summary,
                       EHEFirstGuessParticleName = "PoleEHEOpheliaParticle_ImpLF",
                       EHEFirstGuessName = "PoleEHEOphelia_ImpLF",
                       Looser = True, # For Heartbeat ~ 4 events / day
                       #Loosest = True, # For PnF testing ~ 4 events / hour
                       If = (If and (lambda f: 'PoleEHEOphelia_ImpLF' in f) # First Guess can fail
                             and (lambda f:RunEHEAlertFilter(f)))
                       ) 
