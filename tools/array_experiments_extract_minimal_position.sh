#!/bin/bash
#SBATCH -n 2
#SBATCH --array=11-16
#SBATCH --job-name=minimal
#SBATCH --mem=8GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 4:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

hostname

cd /om/user/sanjanas/minimal-cifar/

counter=0
while [ $counter -le 4 ]
do
    /om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'
    singularity exec -B /om:/om --nv /om/user/xboix/share/localtensorflow.img \
    python /om/user/sanjanas/minimal-cifar/extract_minimal_position.py ${SLURM_ARRAY_TASK_ID} $counter
    echo $counter
    ((counter++))
done
