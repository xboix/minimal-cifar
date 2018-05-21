#!/bin/bash
#SBATCH -n 2
#SBATCH --array=3,8,13,18,23
#SBATCH --job-name=minimal
#SBATCH --mem=16GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


hostname


cd /om/user/xboix/src/minimal-cifar/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/extract_minimal.py ${SLURM_ARRAY_TASK_ID} 2


