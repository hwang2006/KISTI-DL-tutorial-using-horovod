#!/bin/sh
#SBATCH -J pytorch_horovod # job name
#SBATCH --time=24:00:00 # walltime 
#SBATCH --comment=pytorch # application name
#SBATCH -p amd_a100nv_8 # partition name (queue or class)
#SBATCH --nodes=4 # the number of nodes
#SBATCH --ntasks-per-node=8 # number of tasks per node
#SBATCH --cpus-per-task=4 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:8 # number of GPUs per node

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1

##srun python ./train_hvd.py
srun python ./tf_keras_fashion_mnist.py
