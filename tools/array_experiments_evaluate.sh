#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1-13
#SBATCH --job-name=minimal
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/


hostname

cd /cbcl/cbcl01/xboix/src/minimal-cifar/


singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/test_minimal.py ${SLURM_ARRAY_TASK_ID}


