import os

class config:

    def __init__(self):

        self.wins = "submit_2.dat"

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
#SBATCH -J PMF_{0}
#SBATCH -p urgent
#SBATCH -A urgent
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
source /public/software/apps/gromacs/2022.2/bin/GMXRC.bash
export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64

# Assign OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd prod
gmx mdrun -s prod.tpr -cpi prod.cpt -deffnm prod -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf prodf.xvg -px prodx.xvg -noappend -nsteps 7500000  
'''.format(i.replace("win_", ""))
        )

def main():

    settings = config()  
    submit(settings.wins)

if __name__ == '__main__':
    main()