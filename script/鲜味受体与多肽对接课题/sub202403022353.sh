#!/bin/bash
#SBATCH -J aaa
#SBATCH -p urgent
#SBATCH -A urgent
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1

date

echo "Start time: $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job directory: $(pwd)"
echo "CPUS_used: $SLURM_CPUS_PER_TASK"


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


export PATH="/public/home/yqyang/software/Miniconda3/envs/colab/bin:$PATH"

colabfold_batch --num-models 5 --num-recycle 10 --amber --model-type alphafold2_multimer_v3 test0104_1b662.a3m result/
