#!/bin/bash

#SBATCH --job-name=thg_long # create a short name for your job
#SBATCH --account=pi-dachxiu
#SBATCH --partition=caslake
#SBATCH --ntasks=90       # 1 CPU core to drive GPU
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G  # NOTE DO NOT USE THE --mem= OPTION 
#SBATCH --output=zcpu.out
#SBATCH --error=zcpu.err
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


module load parallel


srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog zcpu_run.log --resume"

# $parallel "$srun ./single_run.sh {1} {2}" ::: ai_163_sin.yaml ai_163_hot.yaml dqn_163_sin.yaml dqn_163_hot.yaml  ai_463_sin.yaml  ai_463_hot.yaml ::: {1..15}
$parallel "$srun ./single_run.sh {1} {2}" :::  ai_1123_sin.yaml ai_1123_hot.yaml ai_4123_sin.yaml  ai_4123_hot.yaml ::: {1..15}