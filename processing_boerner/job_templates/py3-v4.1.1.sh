#!/bin/bash
#PBS -l nodes=1:ppn={cpus}
#PBS -l pmem={memory}
#PBS -l mem={memory}
#PBS -l vmem={memory}
#PBS -l pvmem={memory}
#PBS -l walltime={walltime}
#PBS -o {processing_folder}/logs/{step_name}_run_{run_number}_${PBS_JOBID}.out
#PBS -e {processing_folder}/logs/{step_name}_run_{run_number}_${PBS_JOBID}.err
#PBS -q long
#PBS -S /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
FINAL_OUT={final_out}
KEEP_CRASHED_FILES={keep_crashed_files}
WRITE_HDF5={write_hdf5}
WRITE_I3={write_i3}


echo 'Starting job on Host: '$HOSTNAME
echo 'Loading py3-v4.1.1'
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
export PYTHONUSERBASE={python_user_base}
echo 'Using PYTHONUSERBASE: '${PYTHONUSERBASE}

export MPLBACKEND=agg
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.7/site-packages:$PYTHONPATH


echo $FINAL_OUT
if [ -z ${PBS_JOBID} ] && [ -z ${_CONDOR_SCRATCH_DIR} ]
then
    echo 'Running Script w/o temporary scratch'
    {script_folder}/scripts/{script_name} {yaml_copy} {run_number} --no-scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -ne 0 ] && [ $KEEP_CRASHED_FILES -eq 0 ] ; then
        echo 'Deleting partially processed file! ' $FINAL_OUT
        rm ${FINAL_OUT}.i3.bz2
        rm ${FINAL_OUT}.hdf5
    fi
else
    echo 'Running Script w/ temporary scratch'
    if [ -z ${_CONDOR_SCRATCH_DIR} ]
    then
        cd /scratch/${USER}
    else
        cd ${_CONDOR_SCRATCH_DIR}
    fi
    {script_folder}/scripts/{script_name} {yaml_copy} {run_number} --scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -eq 0 ] || [ $KEEP_CRASHED_FILES -eq 1 ]; then
        if [ "$WRITE_HDF5" = "True" ]; then
            cp *.hdf5 {output_folder}
        fi
        if [ "$WRITE_I3" = "True" ]; then
            cp *.i3.bz2 {output_folder}
        fi
    fi

    # Clean Up
    if [ "$WRITE_HDF5" = "True" ]; then
        rm *.hdf5
    fi
    if [ "$WRITE_I3" = "True" ]; then
        rm *.i3.bz2
    fi
fi
exit $ICETRAY_RC

