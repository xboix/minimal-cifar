#!/bin/bash

srun -t 2:00:00 --qos=cbmm \
singularity exec -B /om:/om /om/user/xboix/singularity/xboix-singularity-tensorflow.img tensorboard \
--port=6058 \
--logdir=/om/user/xboix/src/robustness/log/
