#!/bin/bash
#SBATCH -n 2
#SBATCH --array=7-882
#SBATCH --job-name=robustness
#SBATCH --mem=32GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/


cd /om/user/xboix/src/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/robustness/main.py ${SLURM_ARRAY_TASK_ID}


