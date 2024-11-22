import os
import shutil

class config:

    def __init__(self):

        self.reaction_grp = "PGL-phe"
        self.target_grp = "pmf-target"
        self.time = float(500)   # ns
        self.wins = "submit_1.dat"
        self.queue = "urgent"
        self.name = "PMF"

def generate_mdp(reaction_grp, target_grp, time):

    if os.path.exists(os.path.join(".", "mdp")):
        shutil.rmtree(os.path.join(".", "mdp"))
        os.makedirs(os.path.join(".", "mdp"))
    else:
        os.makedirs(os.path.join(".", "mdp"))

    nsteps = str(int(time/0.000002))

    prod_mdp = open(os.path.join(".", "mdp", "prod.mdp"), "w")
    prod_mdp.write(
'''title       = Umbrella pulling simulation
;define      = -DPOSRES_pro
; Run parameters
integrator  = md
dt          = 0.002
tinit       = 0
nsteps      = {0}
nstcomm     = 100
; Output parameters
nstxout     = 0      ; every 10 ps
nstvout     = 0
nstfout     = 0
nstxtcout   = 25000       ; every 1 ps
nstenergy   = 250000
; Bond parameters
constraint_algorithm    = lincs
constraints             = h-bonds
continuation            = yes       ; continuing from NPT
; Single-range cutoff scheme
cutoff-scheme   = Verlet
nstlist         = 20
ns_type         = grid
rlist           = 1.4
rcoulomb        = 1.4
rvdw            = 1.4
; PME electrostatics parameters
coulombtype     = PME
fourierspacing  = 0.12
fourier_nx      = 0
fourier_ny      = 0
fourier_nz      = 0
pme_order       = 4
ewald_rtol      = 1e-5
optimize_fft    = yes
; Berendsen temperature coupling is on in two groups
tcoupl          = Nose-Hoover
tc_grps         = SOLU MEMB SOLV
tau_t           = 1.0 1.0 1.0
ref_t           = 303.15 303.15 303.15
; Pressure coupling is on
pcoupl          = Parrinello-Rahman
pcoupltype      = semiisotropic
tau_p           = 5.0
compressibility = 4.5e-5  4.5e-5
ref_p           = 1.0     1.0
refcoord_scaling = com
; Generate velocities is off
gen_vel     = no
; Periodic boundary conditions are on in all directions
pbc     = xyz
; Long-range dispersion correction
DispCorr    = EnerPres
; Pull code
pull                    = yes
pull_ncoords            = 1         ; only one reaction coordinate
pull_ngroups            = 2         ; two groups defining one reaction coordinate
pull_group1_name        = {1}
pull_group2_name        = {2}
pull_coord1_type        = umbrella  ; harmonic potential
pull_coord1_geometry    = distance  ; simple distance increase
pull_coord1_dim         = Y Y Y
pull_coord1_groups      = 1 2
pull_coord1_start       = yes       ; define initial COM distance > 0
pull_coord1_rate        = 0      ; 0.01 nm per ps = 10 nm per ns
pull_coord1_k           = 1000      ; kJ mol^-1 nm^-2
;pull-group1-pbcatom     = 1394
;pull-pbc-ref-prev-step-com = yes
'''.format(nsteps, reaction_grp, target_grp)
    )

def generate_submit_sh(name, queue):

    submitsh = open(os.path.join(".", "job.sh"), "w")
    submitsh.write(
'''#!/bin/bash
#SBATCH -J {0}
#SBATCH -p {1}
#SBATCH -A {1}
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
source /public/software/apps/gromacs/2022.2/bin/GMXRC.bash
export LD_LIBRARY_PATH=/public/software/lib/:$LD_LIBRARY_PATH
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64

# Assign OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

############################
mkdir npt
cd npt
if [ ! -f npt.gro ]; then
    gmx grompp -f ../mdp/npt.mdp -c ../nvt/nvt.gro -t ../nvt/nvt.cpt -p ../topol.top -o npt.tpr -r ../nvt/nvt.gro -maxwarn 4 -n ../index.ndx
    gmx mdrun -s npt.tpr -deffnm npt -ntmpi 1 -nb gpu -bonded gpu -gpu_id 0 -pme gpu
fi
#########################
mkdir ../pull
cd ../pull
if [ ! -f pull.gro ];then
    gmx grompp -f ../mdp/pull.mdp -c ../npt/npt.gro -t ../npt/npt.cpt -p ../topol.top -o pull.tpr -r ../npt/npt.gro -maxwarn 4 -n ../index.ndx
    gmx mdrun -s pull.tpr -deffnm pull -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf pullf.xvg -px pullx.xvg
fi         
cd ..                                                                                                                            
'''.format(name, queue)
    )

def submit(wins):

    windows = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        windows.append(i.replace("\n", ""))

    for i in windows:
        job = open(os.path.join(".", i, "job.sh"), "w")
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

#########################
mkdir prod
cd prod
if [ ! -f prod.gro ];then
    gmx grompp -f ../../mdp/prod.mdp -c ../win_{0}.gro -p ../../topol.top -o prod.tpr -r ../win_{0}.gro -maxwarn 4 -n ../../index.ndx
    gmx mdrun -s prod.tpr -deffnm prod -dhdl dhdl -ntmpi 1 -nb gpu -bonded gpu -pme gpu -gpu_id 0 -pf prodf.xvg -px prodx.xvg -nsteps 10000000
fi
'''.format(i.replace("win_", ""))
        )

def main():

    settings = config()  
    generate_mdp(settings.reaction_grp, settings.target_grp, settings.time)   
    # generate_submit_sh(settings.name, settings.queue)    
    submit(settings.wins)

if __name__ == '__main__':
    main()