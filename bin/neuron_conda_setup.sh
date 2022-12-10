#!/bin/bash

USER=$(whoami)
WORK_DIR=/scratch/$USER
CONDA_DIR=$WORK_DIR/miniconda3
BASH_FILE=~/.bashrc

if [[ $(grep "conda initialize" $BASH_FILE) ]] ; then
    echo "CONDA setup already initialized"
    return 0
else
    # conda init
    /apps/applications/miniconda3/bin/conda init

    # export CONDA ENVS and PKGS PATH environment
    export CONDA_ENVS_PATH=$CONDA_DIR/envs
    export CONDA_PKGS_DIRS=$CONDA_DIR/pkgs    
fi

# set conda path
source $BASH_FILE

conda config --set auto_activate_base False 
