#!/bin/bash

# conda init
/apps/applications/miniconda3/bin/conda init

USER=$(whoami)
WORK_DIR=/scratch/$USER
CONDA_DIR=$WORK_DIR/miniconda3

# export CONDA ENVS and PKGS
export CONDA_ENVS_PATH=$CONDA_DIR/envs
export export CONDA_PKGS_DIRS=$CONDA_DIR/pkgs

# set conda path
source ~/.bashrc

conda config --set auto_activate_base False 
