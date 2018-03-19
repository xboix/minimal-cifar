#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0-6
#SBATCH --job-name=robustness
#SBATCH --mem=80GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/xboix/src/robustness/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/robustness/main.py ${SLURM_ARRAY_TASK_ID}


