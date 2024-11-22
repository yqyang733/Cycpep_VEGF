#!/bin/bash
#SBATCH -J smd
#SBATCH -p multi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1

echo "Start time: $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job directory: $(pwd)"

# Decide the software version
source /public/software/profile.d/apps_gromacs_2023.2.sh

# Assign OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

############################
mkdir pull
cd pull
if [ ! -f pull.gro ];then
    gmx grompp -f ../mdp/pull.mdp -c ../frame.gro -p ../topol.top -o pull.tpr -r ../frame.gro -maxwarn 4 -n ../index.ndx
    gmx mdrun -s pull.tpr -deffnm pull -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf pullf.xvg -px pullx.xvg
fi         
cd ..                                                                                                                            
