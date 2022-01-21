#!/bin/bash

#SBATCH --job-name=thg # create a short name for your job
#SBATCH --account=pi-dachxiu
#SBATCH --partition=caslake
#SBATCH --ntasks=120       # 1 CPU core to drive GPU
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G  # NOTE DO NOT USE THE --mem= OPTION 
#SBATCH --time=15:00:00          # total run time limit (HH:MM:SS)

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"


module load parallel


srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog thg_run.log --resume"

$parallel "$srun ./single_run.sh {1} {2}" :::  ai_hot_1_1_20_3.yaml ai_hot_1_4_20_3.yaml ai_hot_3_1_20_3.yaml ai_hot_3_4_20_3.yaml ai_sin_1_1_20_3.yaml ai_sin_1_4_20_3.yaml ai_sin_3_1_20_3.yaml ai_sin_3_4_20_3.yaml ::: {1..15}