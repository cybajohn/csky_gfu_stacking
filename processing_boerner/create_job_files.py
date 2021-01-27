#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import stat
import string
import glob

import click
import yaml
import getpass
import itertools

from batch_processing import create_pbs_files, create_dagman_files

try:
    from good_run_list_utils import get_exp_dataset_jobs
except ImportError as e:
    print(e)
    print('Continuing without exp dataset support.')

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


ESCAPE_CHARS = ['=', ' ', '\\']


def escape(file_path):
    """Escape characters from a file path. Inverse of uescape().

    Parameters
    ----------
    file_path : str
        The file path that should be escaped

    Returns
    -------
    str
        The file path with characters escaped.
    """
    for escape_char in ESCAPE_CHARS:
        file_path = file_path.replace(escape_char, '\\'+escape_char)
    return file_path


def unescape(file_path):
    """*Unescape* characters from a file path. Inverse of escape().

    Parameters
    ----------
    file_path : str
        The file path that should be uescaped

    Returns
    -------
    str
        The file path with characters uescaped.
    """
    for escape_char in ESCAPE_CHARS:
        file_path = file_path.replace('\\'+escape_char, escape_char)
    return file_path


def retrieve_existing_runs(param_dict, runs):
    """ Extract run numbers from existing files

    Parameters
    ----------
    param_dict : dict
        config of the specific dataset param

    Returns
    -------
    list
        list of the existing run numbers
    """
    folder_num_min = min(runs)//1000
    folder_num_max = max(runs)//1000
    folder_num_list = [idx for idx in
                        range(folder_num_min, folder_num_max)]
    folder_num_list.append(folder_num_max)

    existing_runs = []
    cfg = SafeDict()
    cfg.update(param_dict)
    for folder_num in folder_num_list:
        cfg['folder_num_pre_offset'] = folder_num
        cfg['folder_pattern'] = param_dict['folder_pattern'].format(**cfg)
        run_folder = os.path.dirname(
                    param_dict['in_file_pattern']).format(**cfg)
        files = sorted(glob.glob('{}/*.i3*'.format(run_folder)))
        runs_str = [idx.split('.')[-3] for idx in files
                    if 'GeoCalibDetectorStatus' not in idx]
        runs_list = list(map(int, runs_str))
        existing_runs.extend(runs_list)

    return existing_runs


def write_job_files(config, check_existing=False):
    with open(config['job_template']) as f:
        template = f.read()

    scripts = []
    run_numbers = []

    # go through all datasets defined in config
    for dataset in config['datasets']:
        print('Now creating job files for dataset: {}'.format(dataset))
        dataset_dict = SafeDict(config['datasets'][dataset])
        dataset_dict['dataset_name'] = dataset

        # create dummy cycler if none is provided
        if 'cycler' not in dataset_dict:
            dataset_dict['cycler'] = {'dummy': [True]}

        # create list of parameters to cycle through
        param_names = []
        param_values_list = []
        for cycle_param in dataset_dict['cycler']:
            param_names.append(cycle_param)
            param_values_list.append(dataset_dict['cycler'][cycle_param])

        cycler_counter = 0
        # Now go through each configuration and
        # crate job files
        for param_values in itertools.product(*param_values_list):

            # create param_dict for this set of configurations
            param_dict = SafeDict()

            # copy settings from main config
            # and general dataset config
            # into this dictionary
            param_dict.update(config)
            param_dict.update(dataset_dict)

            # now update parameters from cycler
            for name, value in zip(param_names, param_values):
                param_dict[name] = value

            # Check if this is an experimental data dataset and get run numbers
            if 'exp_dataset_run_glob' in param_dict:
                runs, param_dict = get_exp_dataset_jobs(param_dict)
            elif 'runs_list' in param_dict:
                # use the list of runs as specified in the config
                runs = dataset_dict['runs_list']
            else:
                # get a list of run numbers to process
                runs = range(*dataset_dict['runs_range'])

            # ignore certain runs if these are provided:
            if 'runs_to_ignore' in dataset_dict:
                runs = [r for r in runs
                        if r not in dataset_dict['runs_to_ignore']]

            if 'check_existing_runs' in dataset_dict:
                if dataset_dict['check_existing_runs']:
                    existing_runs = retrieve_existing_runs(param_dict, runs)
                    runs = [r for r in runs if r in existing_runs]

            # check if CPU or GPU
            if param_dict['resources']['gpus'] == 0:
                param_dict['python_user_base'] = param_dict[
                                                        'python_user_base_cpu']
            elif param_dict['resources']['gpus'] == 1:
                param_dict['python_user_base'] = param_dict[
                                                        'python_user_base_gpu']
            else:
                raise ValueError('More than 1 GPU is currently not supported!')

            # create processing folder
            param_dict['processing_folder'] = unescape(os.path.join(
                        param_dict['data_folder']+'/processing',
                        param_dict['out_dir_pattern'].format(**param_dict)))

            jobs_output_base = os.path.join(param_dict['processing_folder'],
                                            'jobs')

            if not os.path.isdir(jobs_output_base):
                os.makedirs(jobs_output_base)
            log_dir_base = os.path.join(param_dict['processing_folder'],
                                        'logs')
            if not os.path.isdir(log_dir_base):
                os.makedirs(log_dir_base)

            # update config and save individual
            # config for each dataset
            param_dict['scratchfile_pattern'] = unescape(os.path.basename(
                                            param_dict['out_file_pattern']))

            found_unused_file_name = False
            while not found_unused_file_name:
                filled_yaml = unescape(
                    '{config_base}_{cycler_counter:04d}'.format(
                        config_base=os.path.join(
                                    param_dict['processing_folder'],
                                    param_dict['config_base_name']),
                        cycler_counter=cycler_counter))
                if os.path.exists(filled_yaml):
                    # there is already a config file here, so increase counter
                    cycler_counter += 1
                else:
                    found_unused_file_name = True

            cycler_counter += 1
            param_dict['yaml_copy'] = filled_yaml
            with open(param_dict['yaml_copy'], 'w') as yaml_copy:
                yaml.dump(dict(param_dict), yaml_copy,
                          default_flow_style=False)

            # iterate through runs
            completed_run_folders = []
            for run_num in runs:

                # create variables that can be used
                param_dict['run_number'] = run_num
                param_dict['folder_num_pre_offset'] = run_num//1000
                param_dict['folder_num'] = (param_dict['folder_offset'] +
                                            run_num//1000)

                param_dict['run_folder'] = param_dict['folder_pattern'].format(
                                                                **param_dict)

                # fill final output file string
                final_out = unescape(os.path.join(
                        param_dict['data_folder'],
                        param_dict['out_dir_pattern'].format(**param_dict)))

                if param_dict['merge_files']:
                    param_dict['log_dir'] = log_dir_base
                    jobs_output = jobs_output_base
                else:
                    # crate sub directory for logs
                    param_dict['log_dir'] = os.path.join(
                                log_dir_base,
                                param_dict['run_folder'].format(**param_dict))

                    if not os.path.isdir(param_dict['log_dir']):
                        os.makedirs(param_dict['log_dir'])

                    # create sub directory for jobs
                    jobs_output = os.path.join(
                                jobs_output_base,
                                param_dict['run_folder'].format(**param_dict))
                    if not os.path.isdir(jobs_output):
                        os.makedirs(jobs_output)

                    final_out = os.path.join(
                                final_out,
                                param_dict['run_folder'].format(**param_dict))

                final_out = os.path.join(final_out,
                                         param_dict['out_file_pattern'].format(
                                                                **param_dict))

                param_dict['final_out'] = unescape(final_out)

                if param_dict['merge_files']:
                    if param_dict['run_folder'] in completed_run_folders:
                        # skip if the folder has already been taken care of
                        continue
                    else:
                        # remember which folders have been taken care of
                        completed_run_folders.append(param_dict['run_folder'])

                if check_existing:

                    # Assume files already exist
                    already_exists = True

                    # Does the hdf5 file already exist?
                    if param_dict['write_hdf5']:
                        if not os.path.isfile('{}.hdf5'.format(param_dict[
                                                                'final_out'])):
                            already_exists = False

                    # Does the i3 file already exist?
                    if param_dict['write_i3']:
                        if not os.path.isfile('{}.i3.bz2'.format(param_dict[
                                                                'final_out'])):
                            already_exists = False

                    # The files which are to be written,
                    # already exist. So skip these files
                    if already_exists:
                        continue

                output_folder = os.path.dirname(final_out)
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                param_dict['output_folder'] = output_folder
                file_config = string.Formatter().vformat(template, (),
                                                         param_dict)
                script_name = 'job_{final_out_base}.sh'.format(
                    final_out_base=os.path.basename(param_dict['final_out']),
                    **param_dict)
                script_path = os.path.join(jobs_output, script_name)
                with open(script_path, 'w') as f:
                    f.write(file_config)
                st = os.stat(script_path)
                os.chmod(script_path, st.st_mode | stat.S_IEXEC)
                scripts.append(script_path)
                run_numbers.append(run_num)
    return scripts, run_numbers


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--data_folder', '-d', default=None,
              help='folder were all files should be placed')
@click.option('--processing_scratch', '-p', default=None,
              help='Folder for the DAGMAN Files')
@click.option('--dagman/--no-dagman', default=False,
              help='Write/Not write files to start dagman process.')
@click.option('--pbs/--no-pbs', default=False,
              help='Write/Not write files to start processing on a pbs system')
@click.option('--resume/--no-resume', default=False,
              help='Resume processing -> check for existing output')
def main(data_folder,
         config_file,
         processing_scratch,
         pbs,
         dagman,
         resume):
    config_file = click.format_filename(config_file)
    with open(config_file, 'r') as stream:
        config = SafeDict(yaml.full_load(stream))
    config['script_folder'] = SCRIPT_FOLDER
    config['config_base_name'] = os.path.basename(os.path.join(config_file))

    if 'data_folder' in config:
        data_folder = config['data_folder']
        print('Found "data_folder" variable in config.\n'
              'Adjusting data output path to:\n\t{}'.format(data_folder))

    if data_folder is None:
        default = '/data/user/{}/processing_default/data/'.format(getpass.getuser())
        data_folder = click.prompt(
            'Please enter the dir were the files should be stored:',
            default=default)
    data_folder = os.path.abspath(data_folder)
    if data_folder.endswith('/'):
        data_folder = data_folder[:-1]
    config['data_folder'] = data_folder

    if dagman or pbs:
        if processing_scratch is None:
            default = '/scratch/{}/processing_default'.format(
                getpass.getuser())
            processing_scratch = click.prompt(
                'Please enter a processing scratch:',
                default=default)
        config['processing_scratch'] = os.path.abspath(processing_scratch)

    script_files, run_numbers = write_job_files(config, check_existing=resume)

    if dagman or pbs:

        scratch_subfolder = '{}_{}'.format(
                            config['script_name'].replace('.py', ''),
                            config['config_base_name'].replace('.yaml', ''))
        scratch_folder = os.path.join(config['processing_scratch'],
                                      scratch_subfolder)

        if not os.path.isdir(scratch_folder):
            os.makedirs(scratch_folder)
        if dagman:
            create_dagman_files(config,
                                script_files,
                                run_numbers,
                                scratch_folder)
        if pbs:
            create_pbs_files(config,
                             script_files,
                             run_numbers,
                             scratch_folder)


if __name__ == '__main__':
    main()
