#!/bin/bash

# make cuda
module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 cmake/3.16.9

# create the horovd virtual environment
# make sure that miniconda was installed in /scratch/userID/miniconda3
# prefix (=horovod directory): /scratch/userID/miniconda3/envs/hvd-env
conda env create -f hvd-envs.yml --force

# activate the horovod environment
conda activate hvd-env 


# install horovod in the horovod environment
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod==0.26.1

