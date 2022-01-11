#!/bin/bash

module unload pytorch
module unload cuda
module unload python

module load python  

conda init bash
# module load cuda/11.5
module load pytorch/1.10
conda activate thg


echo task $1 $2 seq:$PARALLEL_SEQ host:$(hostname) date:$(date)

python3 train_agents.py --config-file configs/$1




