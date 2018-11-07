#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=minimal
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


hostname

cd /om/user/sanjanas/minimal-cifar/

counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/sanjanas/minimal-cifar/extract_minimal.py ${SLURM_ARRAY_TASK_ID} $counter
    echo $counter
    ((counter++))
done
