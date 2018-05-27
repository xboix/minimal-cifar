#!/bin/bash
#SBATCH -n 2
#SBATCH --array=13-17
#SBATCH --job-name=minimal
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH --workdir=./log/

cd /cbcl/cbcl01/xboix/src/minimal-cifar/

singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/main.py ${SLURM_ARRAY_TASK_ID}


