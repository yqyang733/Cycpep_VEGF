import sys

def gen_job(wins, step):

    job_lst = [(i, i+step-1) for i in range(0, wins, step)]

    for i in range(len(job_lst)):
        rt = open("job_"+str(i)+".sh", "w")
        rt.write(
"""#!/bin/bash
#SBATCH -J {0}_{1}
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

source /public/software/apps/gromacs/2022.2/bin/GMXRC.bash
export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for i in `seq {0} {1}`
do
cd lambda${{i}}
#############################
mkdir em
cd em
if [ ! -f em.tpr ]; then
    gmx grompp -f ../em.mdp -c ../../../ions.gro -p ../../../newtop.top -r ../../../ions.gro -o em.tpr -maxwarn 2
fi
if [ -f em.tpr ] && [ ! -f em.gro ]; then
    gmx mdrun -s em.tpr -deffnm em -ntmpi 1 -nb gpu -gpu_id 0
    #gmx mdrun -s em.tpr -deffnm em -ntomp 10 -ntmpi 1 -gpu_id 0
fi
###########################
mkdir ../nvt
cd ../nvt
if [ ! -f nvt.tpr ];then
    gmx grompp -f ../nvt.mdp -c ../em/em.gro -p ../../../newtop.top -o nvt.tpr -r ../em/em.gro -maxwarn 4 -n ../../../index.ndx
fi
if [ -f nvt.tpr ] && [ ! -f nvt.gro ];then
    gmx mdrun -s nvt.tpr -deffnm nvt -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0
fi
##########################
mkdir ../npt
cd ../npt
if [ ! -f npt.tpr ];then
    gmx grompp -f ../npt.mdp -c ../nvt/nvt.gro -t ../nvt/nvt.cpt -p ../../../newtop.top -o npt.tpr -r ../nvt/nvt.gro -maxwarn 4 -n ../../../index.ndx
fi
if [ -f npt.tpr ] && [ ! -f npt.gro ];then
    gmx mdrun -s npt.tpr -deffnm npt -ntmpi 1 -nb gpu -bonded gpu -gpu_id 0 -pme gpu
fi
###################################
mkdir ../prod
cd ../prod
if [ ! -f prod.tpr ];then
    gmx grompp -f ../md.mdp -c ../npt/npt.gro -t ../npt/npt.cpt -p ../../../newtop.top -o prod.tpr -r ../npt/npt.gro -maxwarn 4 -n ../../../index.ndx
fi
if [ -f prod.tpr ] && [ ! -f prod.gro ];then
    gmx mdrun -s prod.tpr -deffnm prod -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -gpu_id 0 -pme gpu
fi
###################################
cd ../..
done
""".format(str(job_lst[i][0]), str(job_lst[i][1]))
        )
        rt.close()

def main():

    wins = int(sys.argv[1])
    step = int(sys.argv[2])
    gen_job(wins, step)

if __name__=="__main__":
    main()
