# Distributed deep learning using Horovod on Neuron

This repository is intended to guide users to run his/her distributed deep learning codes on multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Neuron. Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Some thoughts on large-scale distributed training on supercompuer
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

1. Download Anaconda or Miniconda. Miniconda is fast and could be sufficient for distributed deep learning practises. 
```
### Anaconda
$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
### Miniconda
$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Install Miniconda
```
$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3
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
[no] >>> yes
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
$ source ~/.bashrc    # set conda path
$ conda config --set auto_activate_base false
$ which conda
/scratch/$USER/miniconda3/condabin/conda
$ conda --version
conda 4.12.0
```
## Building Horovod
Now you are ready to build Horovod as a conda virtual environment: 
1. load modules 
```
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 cmake/3.16.9
```
2. create a new conda virtual environment and activate the environment.
```
$ conda create -n horovod
$ conda activate horovod
```
3. install the pytorch conda package & the tensorflow pip package
```
(horovod) $ conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
(horovod) $ pip install tensorflow-gpu==2.10.0
```
4. install the horovod pip package with support for tensorflow and pytorch with NCCL, MPI and GLOO enabled
```
(horovod) $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod
```
5. verify the horovod conda environment. You should see something like the following:
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



