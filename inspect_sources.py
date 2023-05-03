import os
import json
from glob import glob
import gzip
import numpy as np
import healpy as hp
import histlite as hl

from IPython import embed

import matplotlib.pyplot as plt

import csky as cy

from _paths import PATHS
import _loader

import math

print("Loading sources")
sources = _loader.easy_source_list_loader()

energy = [s["energy"] for s in sources]

print("energy min: ", np.amin(energy), " max: ", np.amax(energy))

plt.hist(energy, bins=np.logspace(2,np.log10(6000)))
plt.xscale('log')
plt.ylabel(r"number of events")
plt.xlabel(r"energy in TeV")
plt.savefig("test_plots/sources_energy.pdf")
plt.clf()

embed()

ana_dir = os.path.join(PATHS.data, "ana_cache", "sig_time_dep_t0_dt_gamma_ran_new")

cy.CONF['mp_cpus'] = 5

ana11 = cy.get_analysis(cy.selections.repo,
                        'version-004-p00', cy.selections.PSDataSpecs.my_cleaned_data,
                        dir=ana_dir
    )

# load sources

t_max = ana11.mjd_max
t_min = ana11.mjd_min


# Check if sources are inside the analysis time frame
srcs_all = [src for src in sources if src["mjd"] <= t_max and src["mjd"] >= t_min]

if len(srcs_all) < len(sources):
    print("Number of possible sources reduced ({} -> {}) due to analysis time frame".format(len(sources),len(srcs_all)))
    srcs = srcs_all

n_srcs = 10

if n_srcs > len(srcs):
    n_srcs = len(srcs)

signals = [src["signal"] for src in srcs]
signals_all = [src["signal"] for src in sources]
signals_sorted = np.sort(signals)
signals_used = signals_sorted[~(n_srcs-1):]
signals_mask = np.in1d(signals, signals_used)
signals_mask_2 = np.in1d(signals_all,signals_used)

src_id_all = np.reshape(np.argwhere(signals_mask_2 == True), n_srcs)

src_id = np.reshape(np.argwhere(signals_mask==True),n_srcs)

sources_time_dep = [srcs[_src_id] for _src_id in src_id]

print("signals_used")
embed()


# time int

table = ""

for i,src in enumerate(sources[:36]):
    table = table + str(i+1) + " & " + "{:.2f}".format(np.round(src["mjd"],decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2)) + " \\\ "

src_file = open("tables/sources_table.tex", "w")
n = src_file.write(table)
src_file.close()

table = ""

for i,src in enumerate(sources[36:]):
    table = table + str(i+1 + 36) + " & " + "{:.2f}".format(np.round(src["mjd"],decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2)) + " \\\ "

src_file = open("tables/sources_table_2.tex", "w")
n = src_file.write(table)
src_file.close()

table = ""

for i,src in enumerate(sources[:36]):
    if (np.sum(np.array(src_id_all) == i)) == 1:
        table = table + "\\textcolor{red}{" + str(i+1) +"}" + " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(src["mjd"],decimals=2))+"}" + " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2))+"}"+ " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2))+"}" + " & " +"\\textcolor{red}{"+ "{:.2f}".format(np.round(src["signal"]*100,decimals=2))+"}"+ " \\\ "
    else:
        table = table + str(i+1) + " & " + "{:.2f}".format(np.round(src["mjd"],decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2)) + " & " + "{:.2f}".format(np.round(src["signal"]*100,decimals=2)) + " \\\ "

src_file = open("tables/sources_table_v2.tex", "w")
n = src_file.write(table)
src_file.close()

table = ""

for i,src in enumerate(sources[36:]):
    if (np.sum(np.array(src_id_all) == (i+36))) == 1:
        table = table + "\\textcolor{red}{" + str(i+1+36) +"}" + " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(src["mjd"],decimals=2))+"}" + " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2))+"}"+ " & " + "\\textcolor{red}{"+"{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2))+"}" + " & " +"\\textcolor{red}{"+ "{:.2f}".format(np.round(src["signal"]*100,decimals=2))+"}"+ " \\\ "
    else:
        table = table + str(i+1 + 36) + " & " + "{:.2f}".format(np.round(src["mjd"],decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["dec"]),decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(src["ra"]),decimals=2)) + " & " + "{:.2f}".format(np.round(src["signal"]*100,decimals=2)) + " \\\ "

src_file = open("tables/sources_table_v2_2.tex", "w")
n = src_file.write(table)
src_file.close()


#time dep

table = ""

for i,_src in enumerate(sources_time_dep): # src id +2 due to cut due to analysis time frame
    table = table + str(src_id_all[i]+1) + " & " + "{:.2f}".format(np.round(_src["mjd"],decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(_src["dec"]),decimals=2)) + " & " + "{:.2f}".format(np.round(np.rad2deg(_src["ra"]),decimals=2)) + " & " + "{:.2f}".format(np.round(100*_src["signal"],decimals=2))  + " \\\ "

src_file = open("tables/sources_table_time_dep.tex", "w")
n = src_file.write(table)
src_file.close()




