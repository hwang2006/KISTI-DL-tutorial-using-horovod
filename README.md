# Distributed deep learning using Horovod on Neuron

This repository is intended to promote large-scale distributed deep learning practices and guide users to run his/her distributed deep learning codes on multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Neuron. 

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Some motivational thoughts on large-scale distributed training on supercompuer](#some-motivational-thoughts-on-large-scale-distributed-training-on-supercompuer)
* [Distributed DL training practices on supercomputer](#distributed-dl-training-practices-on-supercomputer)
* [Installing Conda](#installing-conda)
* [Why Horovod for distributed DL](#why-horovod-for-distributed-dl)
* [Building Horovod](#building-horovod)
* [Horovod Usage](#horovod-usage)
* [Running Horovod interactively](#running-horovod-interactively)
* [Submitting and Monitoring a Horovod batch job](#submitting-and-monitoring-a-horovod-batch-job)
* [Running Jupyter](#running-jupyter)
* [Why Singularity Container](#why-singularity-container)
* [Running Horovod interactively using Singularity](#running-horovod-interactively-using-singularity)
* [Building a Horovod Singularity Container image on your scratch directory](#building-a-horovod-singularity-container-image-on-your-scratch-directory)
* [Submitting and Monitoring a Horovod batch job using Singularity](#submitting-and-monitoring-a-horovod-batch-job-using-singularity)
* [A glimpse of Running Pytorch Lightning](#a-glimpse-of-running-pytorch-lightning)
* [References](#references)

## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

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
  
## Distributed DL training practices on supercomputer
We may need to set up some ditributed deep learning routines or workflows by which DL researchers and Supercomputer facilities administrators exchange and share ideas and thoughts as to how to develope and run distributed training/inferencing practices on national supercomputing facilites. It might be that distributed deep learning (DL) practices on national supercomputing facilities are not so hard as we think it is, with proper tools, flexible operation & resource management policies and reasonably easy-to-use services available in the hands of DL researchers and developers. 
 
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205277236-bed745ef-c684-4e65-a87d-9819e118eb4a.png" width=550/></p> 

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
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

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 4.12.0
```

## Why Horovod for distributed DL?
Horovod, developed by Uber in 2017, is a distributed deep learning training framework, aimed to make it easy and simple to take a DL code developed with different DL frameworks like Tensorflow and Pytorch and scale it to run across many GPUs. It is designed with having the followings in mind in the first place:
1. (neutral to DL frameworks to be used) Is it possible to make your DL codes run in parallel irrespective of whether you are using Tensorflow, Keras or Pytorch?    
2. (easy to use & codify) How much modification does one have to make to a existing DL code to make it distributed? and how easy is it to run it?
3. (fast to run) How much faster would it run in distributed mode and how easy is it to scale up to run?

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/206370237-8e2f481d-cee3-453d-8ece-ef1ebd60b142.png" width=750/></p> 

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
4. install the horovod pip package with support for tensorflow and pytorch with [NCCL](https://developer.nvidia.com/nccl), [MPI](https://www.open-mpi.org/) and [GLOO](https://pytorch.org/docs/stable/distributed.html) enabled:
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

## Horovod Usage
To use horovod, five steps/lines to be added in your code:
1. Initialize Horovod.
```
# Tensorflow 
import horovod.tensorflow as hvd
hvd.init()

# Keras
import horovod.keras as hvd
hvd.init()

# Pytorch
import horovod.torch as hvd
hvd.init()
```
2. Pin GPU to each worker, making sure each worker to be allocated to each GPU available.
```
# Tensorflow/Keras
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Pytorch
torch.cuda.set_device(hvd.local_rank())
```
3. Adjust learning rate and wrap the optimizer
```
# Tensorflow
opt = tf.optimizers.Adam(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt,…)

# Keras
opt = keras.optimizers.Adadelta(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt,…)

# Pytorch
opt = optim.SGD(model.parameters(), 0.01 * hvd.size())
opt= hvd.DistributedOptimizer(opt, …)
```
4. Broadcast the initial variable states from the masker worker (rank 0)and synchroize state across workers.
```
# Tensorflow/Keras
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# Pytorch
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```
5. Checkpoint on the master worker (rank 0)
```
# Tensorflow/Keras
if hvd.rank() == 0:
  callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))

# Pytorch
if hvd.rank() == 0:
   state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), } 
   torch.save(state, filepath)
```
6. An example code using Pytorch (see the [src](https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/tree/main/src) directory for full example codes): 
```
import torch
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPUs to local rank
torch.cuda.set_device(hvd.local_rank())

# Build model
model = Net()
model.cuda()
opt = optim.SGD(model.parameters())

# Adjust learning rate and wrap the optimizer
opt = optim.SGD(model.parameters(), 0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt, …)

# Broadcast parameters and optimizer state from the master worker (rank 0)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

for epoch in range (100):
   for batch, (data, target) in enumerate(...):
       opt.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target)
       loss.backward()
       opt.step()
   # checkpoint at every epoch
   if hvd.rank() == 0:
      state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), } 
      torch.save(state, filepath)
```

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
# (option 1) run with srun
(horovod) [gpu32]$ srun -n 8 python train_hvd.py
```
```
# (option 2) run with horovodrun
(horovod) [gpu32]$ horovodrun -np 8 -H gpu32:4,gpu33:4 python train_hvd.py
```
```
# (option 3) run with mpirun
(horovod) [gpu32]$ mpirun -np 8 -H gpu32:4,gpu33:4 python train_hvd.py
```
  - to run on two nodes with 2 GPUs each:
```
# (option 1) run with srun
(horovod) [gpu32]$ srun -n 4 python train_hvd.py
```
```
# (option 2) run with horovodrun
(horovod) [gpu32]$ horovodrun -np 4 -H localhost:2,gpu33:2 python train_hvd.py
```
```
# (option 3) run with mpirun
(horovod) [gpu32]$ mpirun -np 4 -H localhost:2,gpu33:2 python train_hvd.py
```
  - to run on the gpu33 with 2 GPUs:
```
# (option 1) run with horovodrun
(horovod) [gpu32]$ horovodrun -np 2 -H gpu33:2 python train_hvd.py
``` 
```
# (option 2) run with horovodrun using gloo collective communications
(horovod) [gpu32]$ horovodrun --gloo -np 2 -H gpu33:2 python train_hvd.py
```
## Submitting and Monitoring a Horovod batch job
1. edit a batch job script running on 2 nodes with 8 GPUs each:
```
[glogin01]$ cat horovod_batch.sh
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
```
2. submit and execute the batch job:
```
[glogin01]$ sbatch horovod_batch.sh
Submitted batch job 169608
```
3. check & monitor the batch job status:
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            169608    amd_a100nv_8   python   qualis  PENDING       0:00 1-00:00:00      2 (Resources)
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            169616    amd_a100nv_8   python   qualis  RUNNING       1:01 1-00:00:00      2 gpu[32,34]
```
## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the horovod-enabled virtual envrionment that you have created as a python kernel.
1. activate the horovod-enabled virtual environment:
```
[glogin01]$ conda activate horovod
```
2. install Jupyter on the virtual environment:
```
(horovod) [glogin01]$ conda install jupyter
(horovod) [glogin01]$ pip install jupyter-tensorboard
```
3. add the virtual environment as a jupyter kernel:
```
(horovod) [glogin01]$ pip install ipykernel 
(horovod) [glogin01]$ python -m ipykernel install --user --name horovod
```
4. check the list of kernels currently installed:
```
(horovod) [glogin01]$ jupyter kernelspec list
Available kernels:
python3       /home01/$USER/.local/share/jupyter/kernels/python3
horovod       /home01/$USER/.local/share/jupyter/kernels/horovod
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(horovod) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
#SBATCH --time=8:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=2   # number of tasks per node
#SBATCH --gres=gpu:2          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the node name and port
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1

echo "execute jupyter"
source ~/.bashrc
conda activate horovod
cd /scratch/$USER     # the root/work directory of the jupyter lab/notebook to be launched
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your user name
echo "end of the job"
```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if a jupyter notebook server is running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a SSH client (e.g., Putty, PowerShell, Command Prompt, etc) on your PC or laptop and log in to Neuron just by copying and pasting the port_forwarding_command:
```
C:\Users\hwang>ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
Password(OTP):
Password:
```
7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888
Password or token: $USER    # your account ID on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 

## Why Singularity Container? 
[Singularity](https://sylabs.io/) is a container platform desinged for HPC environments, as opposed to Docker designed for IT environments.
- Each conatiner is a single image file
- No root owned daemon processes
- Support share/multi-tenant resource environment
- Support HPC hardware (e.g., Infiniband, GPUs)
- Support HPC applications (e.g., MPI)

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/206393495-554bfc0c-218d-4928-9453-7a0e7790a31a.png" width=650/></p> 

## Running Horovod interactively using Singularity
You don't have to bother to deal with all the hassles of the Conda and Horovod, and just request an allocation of available nodes using the Slurm salloc command and run a proper singularity container built on Neuron. That's it!
```
[glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=2 --time=2:00:00 --gres=gpu:4 --comment=python

# to run NVIDIA NGC tensorflow container
[gpu32]$ srun -n 8 singularity exec --nv /apps/applications/singularity_images/ngc/tensorflow_22.03-tf2-py3.sif python KISTI-DL-tutorial-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py

# to run NVIDIA NGC keras container
[gpu32]$ srun -n 8 singularity exec --nv /apps/applications/singularity_images/ngc/tensorflow_22.03-tf2-keras-py3.sif python KISTI-DL-tutorial-using-horovod/src/keras/keras_imagenet_resnet50.py

# to run NVIDIA NGC pytorch container
[gpu32]$ srun -n 8 singularity exec --nv /apps/applications/singularity_images/ngc/pytorch_22.03-hd-py3.sif python KISTI-DL-tutorial-using-horovod/src/pytorch/pytorch_imagenet_resnet50.py
```
## Building a Horovod Singularity Container image on your scratch directory 
You can build your own Horovod Singularity container image with both Tensorflow and Pytorch enabled. In order to build a Singularity container on Neuron, you need to have a fakeroot permission that you can get by requesting it to the system administrator.  
```
# build a Pytorch image from the NGC Docker Container repository 
[glogin01]$ singularity build --fakeroot pytorch_22.11-py3.sif docker://nvcr.io/nvidia/pytorch:22.11-py3

# create a Singularity recipe file
[glogin01]$ cat horovod.def
Bootstrap: localimage
From: pytorch_22.11-py3.sif

%post
echo "PIP installing tensorflow-gpu..."
pip install tensorflow-gpu==2.10.0
pip install filelock # src/pytorch/pt_mnist.py requires the filelock module

echo "PIP Installing Horovod..."
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod

# build a Horovod container image
[glogin01]$ singularity build --fakeroot tensorflow-pytorch-horovod.sif horovod.def
```
You can also build it from the horovod-enabled docker image that I have created in my docker repository.      
```
[glogin01]$ singularity build --fakeroot tensorflow-pytorch-horovod.sif docker://qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13
```
Now, you can run your own Horovod container. 

```
[glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=2 --time=2:00:00 --gres=gpu:4 --comment=python
[gpu32]$ srun -n 8 singularity exec --nv tensorflow-pytorch-horovod.sif python KISTI-DL-tutorial-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py
[gpu32]$ srun -n 8 singularity exec --nv tensorflow-pytorch-horovod.sif python KISTI-DL-tutorial-using-horovod/src/keras/keras_imagenet_resnet50.py
[gpu32]$ srun -n 8 singularity exec --nv tensorflow-pytorch-horovod.sif python KISTI-DL-tutorial-using-horovod/src/pytorch/pytorch_imagenet_resnet50.py
```

## Submitting and Monitoring a Horovod batch job using Singularity 
1. edit a batch job script running on 2 nodes with 8 GPUs each:
```
[glogin01]$ cat singularity_horovod_batch.sh
#!/bin/sh
#SBATCH -J python # job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=pytorch # application name
#SBATCH -p amd_a100nv_8 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=8 # number of tasks per node
#SBATCH --gres=gpu:8 # number of GPUs per node
#SBATCH --cpus-per-task=4 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load singularity/3.9.7

srun singularity exec --nv /apps/applications/singularity_images/ngc/tensorflow_22.03-tf2-py3.sif \
             python KISTI-DL-tutorial-using-horovod/src/tensorflow/tf_keras_fashion_mnist.py
```
2. submit and execute the batch job:
```
# Note: You don't have to activate the conda environment here
[glogin01]$ sbatch singularity_horovod_batch.sh
```
3. check & monitor the batch job status:
```
[glogin01]$ squeue -u $USER
```


## Singularity Directories on Neuron
1. Singularity Container Images directory
```
[glogin01]$ ls /apps/applications/singularity_images/ngc
./                              lammps:29Oct2020-x86_64.sif*            tensorflow:20.08-tf1-py3.sif*
../                             namd:2.12-171025.sif*                   tensorflow:20.09-tf1-py3-old.sif*
caffe:20.03-py3.sif*            namd:2.13-multinode.sif*                tensorflow:20.09-tf1-py3.sif*
gamess:17.09-r2-libcchem.sif*   namd:3.0-alpha3-singlenode-x86_64.sif*  tensorflow:20.09-tf1-py3-slurm.sif*
gromacs:2018.2.sif*             paraview_egl-py3-5.9.0.sif*             tensorflow:20.09-tf2-py3.sif*
gromacs:2020.2.sif*             pytorch:20.09-py3-old.sif*              tensorflow:20.10-tf1-py3.sif*
gromacs:2020.2-x86_64.sif*      pytorch:20.09-py3.sif*                  tensorflow:20.12-tf1-keras-py3.sif*
gromacs:2020.sif*               pytorch:20.12-py3.sif*                  tensorflow:20.12-tf1-py3.sif*
hpc-benchmarks:20.10-hpcg.sif*  pytorch_22.03-hd-py3.sif*               tensorflow:20.12-tf1-py3-slurm.sif*
hpc-benchmarks:20.10-hpl.sif*   pytorch_22.03-hd-py3-v2.sif*            tensorflow:20.12-tf2-py3.sif*
julia:v1.5.0.sif*               pytorch_22.03-py3.sif*                  tensorflow_22.03-tf1-keras-py3.sif*
julia:v2.4.1.sif*               quantum_espresso:v6.6a1.sif*            tensorflow_22.03-tf1-py3.sif*
lammps:15Jun2020-x86_64.sif*    quantum_espresso:v6.7.sif*              tensorflow_22.03-tf2-keras-py3.sif*
lammps:29Oct2020.sif*           tensorflow:20.01-tf1-py3.sif*           tensorflow_22.03-tf2-py3.sif*
```
2. Distributed DL training job scripts directory
```
[glogin01]$ ls /apps/applications/singularity_images/examples
./  ../  01.pytorch.sh*  02.pytorch_horovod.sh*  03.keras_horovod.sh*  04.srun.sh*  horovod/  pytorch/
```
3. Pytorch examples directory
```
[glogin01]$ ls /apps/applications/singularity_images/examples/horovod/examples/pytorch
./  ../  pytorch_imagenet_resnet50.py  pytorch_mnist.py  pytorch_synthetic_benchmark.py
```
4. Imagenet datasets directories
```
# Training datasets directory
[glogin01]$ cd /apps/applications/singularity_images/imagenet/train

# Validation datasets directory
[glogin01]$ cd /apps/applications/singularity_images/imagenet/val
```

## A glimpse of Running Pytorch Lightning
[Pytorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight wrapper or interface on top of PyTorch, which simplifies the implementation of complex deep learning models. It is a PyTorch extension that enables researchers and practitioners to focus more on their research and less on the engineering aspects of deep learning. PyTorch Lightning automates many of the routine tasks, such as distributing training across multiple GPUs, logging, and checkpointing, so that the users can focus on writing high-level code for their models.

We will show a glimpse of how to run a simple pytorch lightning code on multiple nodes interactively.
1. Install Pytorch Lightning:
```
[glogin01]$ conda activate horovod
(horovod) [glogin01]$ pip install lightning
```
2. request allocation of available GPU-nodes:
```
(horovod) [glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=2 --time=8:00:00 --gres=gpu:4 --comment=python
salloc: Granted job allocation 154173
salloc: Waiting for resource configuration
salloc: Nodes gpu[32-33] are ready for job
```
3. load modules and activate the horovod conda environment:
```
[gpu32]$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1
[gpu32]$ $ conda activate horovod
(horovod) [gpu32]$
```
4. run a pytorch lightning code:
- to run on the two nodes with 4 GPUs each. Pytorch Lightning complains and exits with some runtime error messages when using "srun" with the -n or --ntasks options, so you need to use --ntasks-per-node instead.
```
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=4 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2
```
- to run the Bert NSMC (Naver Sentiment Movie Corpus) example in the src/pytorch-lightning directory, you might need to install additional packages (i.e., emoji, soynlp, transformer and pandas) and download the nsmc datasets, for example, using git cloning
```
(horovod) [gpu32]$ pip install emoji==1.7.0 soynlp transformers pandas
(horovod) [gpu32]$ git clone https://github.com/e9t/nsmc  # download the nsmc datasets in the ./nsmc directory
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=4 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2
```
- to run on the two nodes with 2 GPUs each
```
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=2 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2 --devices 2
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=2 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 2
```
- to run on the two nodes with 1 GPU each
```
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=1 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2 --devices 1
(horovod) [gpu32]$ srun -N 2 --ntasks-per-node=1 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 1
```
- to run one node with 4 GPUs
```
(horovod) [gpu32]$ python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py
(horovod) [gpu32]$ python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py
(horovod) [gpu32]$ srun -N 1 --ntasks-per-node=4 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py
```
- to run one node with 2 GPUs
```
(horovod) [gpu32]$ python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --devices 2
(horovod) [gpu32]$ python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --devices 2
(horovod) [gpu32]$ srun -N 1 --ntasks-per-node=2 python KISTI-DL-tutorial-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --devices 2
```

## References
* [Distributed training on perlmulter supercomputer](https://github.com/hwang2006/distributed-training-on-perlmutter-using-horovod)








