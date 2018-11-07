#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=minimal
#SBATCH --mem=4GB
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

hostname

cd /om/user/sanjanas/minimal-cifar/

singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/sanjanas/minimal-cifar/minimal_images_statistics.py ${SLURM_ARRAY_TASK_ID}

#singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
#python /om/user/xboix/src/minimal-cifar/minimal_images_statistics_small.py ${SLURM_ARRAY_TASK_ID}


