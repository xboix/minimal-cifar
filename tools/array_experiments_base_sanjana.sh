#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0
#SBATCH --job-name=minimal
#SBATCH --mem=80GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/sanjanas/minimal-cifar/
singularity exec -B /om:/om --nv /om/user/xboix/share/belledon-tensorflow-keras-master-latest.simg \
python /om/user/sanjanas/minimal-cifar/main.py ${SLURM_ARRAY_TASK_ID}


