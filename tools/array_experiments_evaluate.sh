#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1-15
#SBATCH --job-name=minimal
#SBATCH --mem=8GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log/


hostname

cd /om/user/xboix/src/minimal-cifar/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/xboix/src/minimal-cifar/test_minimal.py ${SLURM_ARRAY_TASK_ID}


