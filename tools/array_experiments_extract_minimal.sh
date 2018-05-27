#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1-17
#SBATCH --job-name=minimal
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --workdir=./log/


hostname

cd /cbcl/cbcl01/xboix/src/minimal-cifar/

counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/xboix/src/minimal-cifar/extract_minimal_small.py ${SLURM_ARRAY_TASK_ID} $counter
    echo $counter
    ((counter++))
done

