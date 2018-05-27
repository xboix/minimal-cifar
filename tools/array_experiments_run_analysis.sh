#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1-14
#SBATCH --job-name=minimal
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH --workdir=./log/

hostname

cd /cbcl/cbcl01/xboix/src/minimal-cifar/

counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/xboix/src/minimal-cifar/extract_minimal.py ${SLURM_ARRAY_TASK_ID} 0
    echo $counter
    ((counter++))
done


counter=0
while [ $counter -le 4 ]
do
    singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
    python /om/user/xboix/src/minimal-cifar/extract_minimal_small.py ${SLURM_ARRAY_TASK_ID} 0
    echo $counter
    ((counter++))
done


singularity exec -B /cbcl/cbcl01/:/om/user/ --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/minimal_images_statistics.py ${SLURM_ARRAY_TASK_ID}


singularity exec -B /om:/om --nv /om/user/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/test_minimal.py ${SLURM_ARRAY_TASK_ID}

