#!/bin/sh
##SBATCH -J python # job name
#SBATCH --time=24:00:00 # walltime 
#SBATCH --comment=pytorch # application name
#SBATCH -p amd_a100nv_8 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=8 # number of tasks per node
#SBATCH --gres=gpu:8 # number of GPUs per node
#SBATCH --cpus-per-task=4 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1
source ~/.bashrc
conda activate horovod

#srun python KISTI-DL-tutorial-using-horovod/src/tensorflow/tf_keras_fashion_mnist.py
srun python tf_keras_fashion_mnist.py
