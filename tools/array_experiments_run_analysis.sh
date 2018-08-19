#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1-16
#SBATCH --job-name=minimal
#SBATCH --mem=8GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 10:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log/

hostname

cd /om/user/xboix/src/minimal-cifar/
/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/test_minimal.py ${SLURM_ARRAY_TASK_ID}

counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/xboix/src/minimal-cifar/extract_minimal.py ${SLURM_ARRAY_TASK_ID} $counter
    echo $counter
    ((counter++))
done


counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/xboix/src/minimal-cifar/extract_minimal_small.py ${SLURM_ARRAY_TASK_ID} $counter
    echo $counter
    ((counter++))
done


singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/minimal_images_statistics.py ${SLURM_ARRAY_TASK_ID}

singularity exec -B /om:/om/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/minimal_images_statistics_small.py ${SLURM_ARRAY_TASK_ID}





