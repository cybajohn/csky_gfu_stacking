# General Processing Framework

This repository is a collection of IceCube processing scripts and configuration files.
The code is adapted from Mathis Börner (https://github.com/mbrner/simulation_scripts)
and its improvement by Mirco Hünnefeld (http://code.icecube.wisc.edu/svn/sandbox/mhuennefeld/processing_scripts/trunk/processing).

It creates jobfiles to run the scripts on the cluster for a dagman system.
There is also the option to create the files for a pbs system, but this is not maintained for years.
Besides one can test the scripts for single jobs locally.

## Usage

```
$ python create_job_files.py "path_to_config_yaml" -d "path_to_store_the_data"
```

check --help for options.

An example config is given in `configs/` folder.

### Example

create dagman files on submitter:
```
USER@submitter$ python create_job_files.py configs/{config_name}.yaml --dagman -d /data/user/${USER}/stopping_muons -p /scratch/${USER}/stopping_muons
```

start dagman:
```
USER@submitter$ /scratch/${USER}/stopping_muons/{script_name}_{config_name}/start_dagman.sh
```

## Requirements

install `click` and `pyyaml` in your virtual env.
how to create a virtual environment within the icecube framework is described [here](https://hackmd.e5.physik.tu-dortmund.de/7KDpq1uFReqR0IPP_KvMog)

## A more detailed description

### Preparations

the main script (in the script folder) should be executable, so e.g.
```
chmod -R ugo+rwx scripts/{script_name}.py
```

### Testing the Scripts locally

One can test the python scripts by creating shell scripts
(of course everything in `/scratch` ;)
```
USER@cobalt$ python create_job_files.py configs/{config_name}.yaml -d /scratch/${USER}/stopping_muons/
```
and execute them locally by hand.
Before executing the shell script, make sure not to be inside an IceTray environment.
```
USER@cobalt$ /scratch/${USER}/stopping_muons/processing/{dataset}/.../{jobfile}.sh
```
The created file can be found in the `/scratch` of the specific cobalt machine.

One can look at the file reentering the IceTray environment with 
```
USER@cobalt$ dataio-shovel /scratch/${USER}/stopping_muons/{dataset}/.../{filename}.i3.bz2
```

### Run Scripts on Cluster

to submit jobs first go to 
```
ssh submitter
```
again create the job files, now with __dagman__ (and of course write log files to `/scratch` ;)
```
USER@submitter$ python create_job_files.py configs/{config_name}.yaml -d /data/user/${USER}/stopping_muons/ -p /scratch/${USER}/stopping_muons --dagman
```
and send the jobs to the cluster, combined into a single submitted job
```
USER@submitter$ /scratch/${USER}/stopping_muons/{script_name}_{config_name}/start_dagman.sh
```
thats it.

### Test Scripts on Cluster

Before sending the files to the cluster, one might just test a small subset, if everything is fine, eg. the first 5 files and see if things run. Therefore just change the `dagman.options` file.
```
cd /scratch/${USER}/stopping_muons/{script_name}_{config_name}/
cp dagman.options dagman.options_test
vim dagman.options_test
10jdG:wq
vim start_dagman.sh
A_test[Esc]:wq
./start_dagman.sh
```
#### resume processing for crashed jobs

If files crashed during processing on the cluster, there is an option to `--resume` the processing and just process the files, that have been crashed.
So ( after fixing the bug ;) just type
```
USER@submitter$ python create_job_files.py configs/{config_name}.yaml -d /data/user/${USER}/stopping_muons/ -p /scratch/${USER}/stopping_muons_resume --dagman --resume
```
the processing folder should be different to the previous processed one.

## Condor Specialties

While processing one can look at the jobs on submitter with `condor_q`.

Further things with condor rescue ...