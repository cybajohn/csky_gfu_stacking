#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT combo/V01-00-02

import os
import timeit

import click
import yaml
import glob

from I3Tray import I3Tray
from icecube import icetray, dataclasses, dataio, hdfwriter

from test_modul_utils import a_new_module

@click.command()
@click.argument('cfg', type=click.Path(exists=True))
@click.argument('run_number', type=int)
@click.option('--scratch/--no-scratch', default=True)
def main(cfg, run_number, scratch):

    start_time = timeit.default_timer()

    with open(cfg, 'r') as stream:
        cfg = yaml.full_load(stream)
    cfg['run_number'] = run_number

    cfg['folder_num_pre_offset'] = cfg['run_number']//1000
    cfg['folder_num'] = cfg['folder_offset'] + cfg['run_number']//1000
    cfg['folder_pattern'] = cfg['folder_pattern'].format(**cfg)
    cfg['run_folder'] = cfg['folder_pattern'].format(**cfg)

    infile = cfg['in_file_pattern'].format(**cfg)

    if cfg['merge_files']:

        input_run_folder = os.path.dirname(infile)
        infiles = [cfg['gcd']]

        # get all files in input run folder
        for name in sorted(glob.glob('{}/*.i3*'.format(input_run_folder))):
            infiles.append(name)

        if len(infiles) <= 1:
            raise IOError('No files found for:\n\t {}/*.i3*'.format(
                                                            input_run_folder))
    else:
        if not cfg['gcd'] is None:
            infiles = [cfg['gcd'], infile]
        else:
            infiles = [infile]

    if scratch:
        outfile = cfg['scratchfile_pattern'].format(**cfg)
        scratch_output_folder = os.path.dirname(outfile)
        if scratch_output_folder and not os.path.isdir(scratch_output_folder):
            os.makedirs(scratch_output_folder)
    else:
        outfile = os.path.join(cfg['data_folder'],
                               cfg['out_dir_pattern'].format(**cfg))
        if not cfg['merge_files']:
                    outfile = os.path.join(outfile,
                                           cfg['run_folder'].format(**cfg))
        outfile = os.path.join(outfile,
                               cfg['out_file_pattern'].format(**cfg))

    tray = I3Tray()

    tray.Add('I3Reader', FilenameList=infiles)



    # --------------------------------------------------
    # The main module that should be processed
    # --------------------------------------------------

    tray.Add(a_new_module, "a_new_module",
        feature_for_module=cfg['feature_for_module'],)



    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    if cfg['write_i3']:
        tray.AddModule("I3Writer", "EventWriter",
                       filename='{}.i3.bz2'.format(outfile),
                       Streams=[icetray.I3Frame.DAQ,
                                icetray.I3Frame.Physics,
                                icetray.I3Frame.TrayInfo,
                                icetray.I3Frame.Simulation,
                                icetray.I3Frame.Stream('S'),
                                ],
                       DropOrphanStreams=[icetray.I3Frame.DAQ],
                       )

    if cfg['write_hdf5']:
        tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
                        Output='{}.hdf5'.format(outfile),
                        CompressionLevel=9,
                        Keys=cfg['HDF_keys'],
                        SubEventStreams=cfg['HDF_SubEventStreams'])
    # --------------------------------------------------

    tray.Execute()

    end_time = timeit.default_timer()
    print('Duration: {:5.3f}s'.format(end_time - start_time))


if __name__ == '__main__':
    main()
