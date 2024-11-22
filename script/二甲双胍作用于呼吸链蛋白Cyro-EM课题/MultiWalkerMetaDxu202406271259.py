import os

class config:

    def __init__(self):

        self.wins = "wins.dat"

def submit(wins):

    windows = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        windows.append(i.replace("\n", ""))

    for i in windows:

        # if not os.path.exists(os.path.join(".", "win_"+str(i), "prod")):
            # os.makedirs(os.path.join(".", "win_"+str(i), "prod"))

        job = open(os.path.join(".", i, "job_1.sh"), "w")
        job.write(
'''#!/bin/bash
#SBATCH -J a_{0}
#SBATCH -p quick
#SBATCH --time=12:00:00
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
#source /public/home/yqyang/software/gmx_2203_plumed-installed_1/bin/GMXRC.bash
#export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
#source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64
#export PATH="/public/home/yqyang/software/mpich-4.1.1-installed/bin:"$PATH
#export PATH="/public/home/yqyang/software/plumed-2.8.1-installed_mpi_1/bin:"$PATH
#export LD_LIBRARY_PATH="/public/home/yqyang/software/mpich-4.1.1-installed/lib:"$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH="/public/home/yqyang/software/plumed-2.8.1-installed_mpi_1/lib:"$LD_LIBRARY_PATH
source /public/software/profile.d/apps_gromacs_2022.5_mpi_plumed.sh

# Assign OMP threads
export OMP_NUM_THREADS=4

cd prod
# gmx mdrun -s prod.tpr -cpi prod.cpt -deffnm prod -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf prodf.xvg -px prodx.xvg -noappend -nsteps 5000000 
gmx_mpi mdrun -s prod.tpr -cpi prod.cpt -deffnm prod -dhdl dhdl -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf prodf.xvg -px prodx.xvg -noappend -nsteps 5000000 -plumed ../plumed.dat 
'''.format(i.replace("win_", ""))
        )

def main():

    settings = config()  
    submit(settings.wins)

if __name__ == '__main__':
    main()
