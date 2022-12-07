# Distributed deep learning using Horovod on Neuron

This repo is intended to guide users to run his/her distributed deep learning codes on multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Neuron. Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Some motivational thoughts on large-scale distributed training on supercompuer
- KISTI-6 with ~600PFs is coming  in 2 years
  - Several thousands of GPUs will be availale to users 
- Is a large-scale LM (Language Model) training a exclusive task that can be carried out by big tech companies (e.g., Google., Meta, MS) running a global data center facilities?
- Why is it that large-scale LM R&D is so hard in Korea? Is is due to:
  - lack of computing resources
  - lack of datasets
  - lack of skills
  - lack of ??
- What can KISTI do in contributing to LM R&D in Korea
  - KISTI is uniquely positioned to running a general-purpose national supercomputing facilities
- Is KISTI's supercomputing facility easy to access for users to do large-scale distributed deep learning R&D?
- What are the most significant barries that prevent users from having access to KISTI supercomputing facities in conducting large-scale distributed trainging and inferencing?
  - Is it because of the traditonal batch-schduling based services?
  - ??
  
## Distributed training workflow on supercomputer
We may need to set up some ditributed deep learning routines or workflows by which DL researchers and Supercomputer facilities administrators exchange and share ideas and thoughts as to how to develope and run distributed training/inferencing practices on national supercomputing facilites. It might be that distributed deep learning (DL) practices on national supercomputing facilities are not so hard as we think it is, with proper tools, flexible operation & resource management policies and reasonably easy-to-use services available in the hands of DL researchers and developers. 
 
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205277236-bed745ef-c684-4e65-a87d-9819e118eb4a.png" width=550/></p> 

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Download Anaconda or Miniconda. Miniconda is fast to install and could be sufficient for distributed deep learning practices. 
```
### (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
### (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Install Miniconda
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

3. finalize installing Miniconda with environment variable set

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 4.12.0
```
## Building Horovod
Now you are ready to build Horovod as a conda virtual environment: 
1. load modules: 
```
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 cmake/3.16.9
```
2. create a new conda virtual environment and activate the environment:
```
[glogin01]$ conda create -n horovod
[glogin01]$ conda activate horovod
```
3. install the pytorch conda package & the tensorflow pip package:
```
(horovod) [glogin01]$ conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
(horovod) [glogin01]$ pip install tensorflow-gpu==2.10.0
```
4. install the horovod pip package with support for tensorflow and pytorch with NCCL, MPI and GLOO enabled:
```
(horovod) [glogin01]$ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod
```
5. verify the horovod conda environment. You should see output something like the following:
```
(horovod) $ horovodrun -cb
Horovod v0.26.1:

Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```
## Why Horovod
Horovod is a distributed deep learning training framework developed by Uber in 2917, aiming to make distributed DL training fast and easy to use. 


## Running Horovod interactively 
Now, you are ready to run distributed training using Horovod on Neuron. 
1. request allocation of available GPU-nodes for interactively running and testing distributed training codes: 
```
(horovod) [glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=2 --time=8:00:00 --gres=gpu:4 --comment=python
salloc: Granted job allocation 154173
salloc: Waiting for resource configuration
salloc: Nodes gpu[32-33] are ready for job
```
In this example case, gpu32 and gpu33 are allocated with 4 GPUs each, and you are residing on the gpu32 node.

2. load modules again on the gpu node:
```
[gpu32]$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 cmake/3.16.9
```
3. activate the horovod conda environment: 
```
[gpu32]$ $ conda activate horovod
(horovod) [gpu32]$
```
4. run & test horovod-enabled distributed DL codes:
  - to run on the two nodes with 4 GPUs each: 
```
### (Option 1)
(horovod) [gpu32]$ srun -n 8 python train_hvd.py
```
```
### (Option 2)
(horovod) [gpu32]$ horovodrun -np 8 -H gpu32:4,gpu33:4 python train_hvd.py
```
```
### (Option 3)
(horovod) [gpu32]$ mpirun -np 8 -H gpu32:4,gpu33:4 python train_hvd.py
```
  - to run on two nodes with 2 GPUs each:
```
### (Option 1)
(horovod) [gpu32]$ srun -n 4 python train_hvd.py
```
```
### (Option 2)
(horovod) [gpu32]$ horovodrun -np 4 -H localhost:2,gpu33:2 python train_hvd.py
```
```
### (Option 3)
(horovod) [gpu32]$ mpirun -np 4 -H localhost:2,gpu33:2 python train_hvd.py
```
  - to run on the gpu33 with 2 GPUs:
```
(horovod) [gpu32]$ horovodrun -np 2 -H gpu33:2  python train_hvd.py
```  
## Submitting & Monitoring a batch job
1. edit a batch job script running on 4 nodes with 8 GPUs each:
```
[glogin01]$ cat ./train_hvd.sh
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

srun python ./train_hvd.py
```
2. to submit and execute the batch job:
```
[glogin01]$ conda activate horovod
(horovod) [glogin01]$ sbatch ./train_hvd.sh
```
3. to check & monitor the batch job status:
```
(horovod) [glogin01]$ squeue -u $USER
```




