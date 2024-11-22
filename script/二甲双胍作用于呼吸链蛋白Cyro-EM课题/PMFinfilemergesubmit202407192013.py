import os
import shutil

class config:

    def __init__(self):

        self.wins = "wins.dat"
        self.step = int(76)

def submit(wins, step):

    windows = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        windows.append(i.replace("\n", ""))

    job_lst = [(i, min((i+step-1), len(windows))) for i in range(0, len(windows), step)]
    
    for i in range(len(job_lst)):
        job_str = " ".join(windows[job_lst[i][0]:(job_lst[i][1]+1)])
        job = open(os.path.join(".", "job_"+str(i)+".sh"), "w")
        job.write(
'''#!/bin/bash
#SBATCH -J a
#SBATCH -p multi
#SBATCH --time=168:00:00
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
source /public/software/profile.d/apps_gromacs_2022.5_mpi_plumed.sh

# Assign OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#########################
for i in {0}
do
cd ${{i}}
mkdir prod
cd prod
if [ ! -f prod.gro ];then
    gmx_mpi grompp -f ../../mdp/prod.mdp -c ../${{i}}.gro -p ../../topol.top -o prod.tpr -r ../${{i}}.gro -maxwarn 4 -n ../../index.ndx
    gmx_mpi mdrun -s prod.tpr -deffnm prod -dhdl dhdl -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf prodf.xvg -px prodx.xvg -nsteps 5000000 -plumed ../plumed.dat
fi
cd ../..
done
'''.format(job_str)
        )

def main():

    settings = config()  
    # generate_mdp(settings.reaction_grp, settings.target_grp, settings.time)   
    # generate_submit_sh(settings.name, settings.queue)    
    submit(settings.wins, settings.step)

if __name__ == '__main__':
    main()