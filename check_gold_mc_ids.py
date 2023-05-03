"""
taken from https://github.com/blaufuss/alert_v2_processing/blob/main/trig_sim_alert_find/process_filtered_sim.py , thanks Erik!
"""


import sys
import numpy as np
import json
import glob

import argparse

from icecube import icetray, dataclasses, dataio
from icecube.realtime_gfu.muon_alerts import gfu_alert_eval
from icecube.realtime_hese.HESE_alerts_v2 import hese_alert_eval
from icecube.realtime_ehe.EHE_alerts_v2 import ehe_alert_eval
"""
parser = argparse.ArgumentParser(description="ehe_stacking")
parser.add_argument("--seed", type=int)
parser.add_argument("--id", type=str)
parser.add_argument("--ntrials", type=int)
args = parser.parse_args()
rnd_seed = args.seed
ntrials = args.ntrials
job_id = args.id
"""

parser = argparse.ArgumentParser(description="ehe_stacking")
parser.add_argument("--infiles", type=str)
parser.add_argument("--gcdfile", type=str)
parser.add_argument("--outfile_1", type=str)

args = parser.parse_args()

in_files = args.infiles.split(",")
gcd_file = args.gcdfile
out_file_1 = args.outfile_1


outfile = out_file_1
infiles = in_files

infiles.sort()

print('Outfile:',outfile)
print('Infile:',infiles)

events_read = 0

run_id = []
event_id = []
mc_one_wt = []
mc_prim_E = []
mc_prim_zen = []
mc_prim_azi = []
gfu_select = []
gfu_gold_select = []
gfu_bronze_select =[]
hese_gold_select = []
hese_bronze_select = []
ehe_gold_select = []

for file in infiles:
    afile = dataio.I3File(file)
    while afile.more():
        events_read += 1
        pass_gfu = False
        pass_hesev2_gold = False
        pass_hesev2_bronze = False
        pass_ehev2_gold = False
        pass_gfu_gold = False
        pass_gfu_bronze = False
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
                pass_gfu = True
                is_alert = gfu_alert_eval(message)
                #if we got an alert, let's disect it
                if is_alert['pass_loose'] or is_alert['pass_tight']:
                    if is_alert['pass_tight']:
                        pass_gfu_gold = True
                    if is_alert['pass_loose']:
                        pass_gfu_bronze = True
                    ## found a gold or bronze
                    #print("********************Found Gold or Bronze")
                    print("GFU:",is_alert)
                #print('GFU')
            if 'HESE' in alert_list:
                #print('HESE')
                is_hese = hese_alert_eval(message)
                if is_hese['pass_tight']:
                    pass_hesev2_gold = True
                if is_hese['pass_loose']:
                    pass_hesev2_bronze = True
                    print("HESE:",is_hese)
            if 'EHE' in alert_list:
                #print('EHE')
                is_ehe = ehe_alert_eval(message)
                if is_ehe['pass_tight']:
                    pass_ehev2_gold = True
                    print("EHE:",is_ehe)
        else:
            print('Failed to find alert list!!')
        if pass_gfu_gold or pass_gfu_bronze or pass_hesev2_gold or \
           pass_hesev2_bronze or pass_ehev2_gold or pass_gfu:
            # Save data for conversion to numpy array later
            e_id = pframe['I3EventHeader'].event_id
            r_id = pframe['I3EventHeader'].run_id
            event_id.append(e_id)
            run_id.append(r_id)
            gfu_select.append(pass_gfu)
            gfu_gold_select.append(pass_gfu_gold)
            gfu_bronze_select.append(pass_gfu_bronze)
            hese_gold_select.append(pass_hesev2_gold)
            hese_bronze_select.append(pass_hesev2_bronze)
            ehe_gold_select.append(pass_ehev2_gold)
            if pframe.Has('I3MCWeightDict'):
                mc_wts = pframe['I3MCWeightDict']
                mc_one_wt.append(mc_wts['OneWeight'])
                mc_prim_E.append(mc_wts['PrimaryNeutrinoEnergy'])
                mc_prim_zen.append(mc_wts['PrimaryNeutrinoZenith'])
                mc_prim_azi.append(mc_wts['PrimaryNeutrinoAzimuth'])
            else:
                print("No Weight Dict!")


#print(mc_one_wt)
print("Events cataloged:",len(mc_one_wt))
print('Processed N_events:',events_read)
arr = np.empty((len(mc_one_wt), ), dtype=[("PassGFU", np.bool),
                                          ("PassGFUGold", np.bool),
                                          ("PassGFUBronze", np.bool),
                                          ("PassHESEGold", np.bool),
                                          ("PassHESEBronze", np.bool),
                                          ("PassEHEGold", np.bool),
                                          ("PrimAzimuth",np.float),
                                          ("PrimZenith",np.float),
                                          ("PrimEnergy", np.float),
                                          ("OneWeight", np.float),
                                          ("RunID", np.int),
                                          ("EventID", np.int)])


arr["PassGFU"] = gfu_select
arr["PassGFUGold"]=gfu_gold_select
arr["PassGFUBronze"]=gfu_bronze_select
arr["PassHESEGold"]=hese_gold_select
arr["PassHESEBronze"]=hese_bronze_select
arr["PassEHEGold"]=ehe_gold_select
arr["PrimAzimuth"]=mc_prim_azi
arr["PrimZenith"]=mc_prim_zen
arr["PrimEnergy"]=mc_prim_E
arr["OneWeight"]=mc_one_wt
arr["RunID"]=run_id
arr["EventID"]=event_id

print("\t{0:6d} events".format(len(arr)))
np.save(outfile, arr)
