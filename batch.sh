#!/bin/bash

#SBATCH --job-name=tiny_hintplay # create a short name for your job
#SBATCH --account=pi-dachxiu
#SBATCH --partition=caslake
#SBATCH --ntasks=80       # 1 CPU core to drive GPU
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G  # NOTE DO NOT USE THE --mem= OPTION 
#SBATCH --output=cpu1.out
#SBATCH --error=cpu1.err
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mma3@chicagobooth.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


# module unload python
# module unload pytorch
# module unload cuda
module load parallel
# module load python/anaconda-2021.05        
# module load pytorch/1.10
# conda init bash
# module load cuda/11.2  
# conda activate thg


srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog cpu_run.log --resume"

$parallel "$srun ./single_run.sh {1} {2}" ::: ai_124_sin.yaml ai_124_hot.yaml dqn_124_sin.yaml dqn_124_hot.yaml ::: {1..20}
