import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob

path_to_fits = os.path.join("/data", "ana", "realtime", "alert_catalog_v2", "fits_files")

#fit_folder = [os.path.join(path_to_fits, ".")]

#fit_files = list([f for f in glob.glob(fit_folder + "/*.fits.gz")])

fit_files = list(glob.glob(os.path.join(path_to_fits,'*.*')))

file_count = len(fit_files)

print("located {} files".format(file_count))

fit_files = [fit_files[0]]

mom_count = 0

gfu_gold_count = 0
signal = []

for ffile in fit_files:
	mom_count+=1
	print('\r', str(mom_count), end = ' of {}'.format(file_count))
	skymap, header = hp.read_map(ffile,h=True, verbose=False)
	header = dict(header)
	from IPython import embed
	embed()
	signal.append(header['SIGNAL'])
	I3type = header['I3TYPE']
	if I3type == 'gfu-gold':
		gfu_gold_count +=1
print('\n')
print(gfu_gold_count)

from IPython import embed
embed()

