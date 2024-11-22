#!/bin/bash
#SBATCH -J top2mmd
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
source /public/software/profile.d/apps_gromacs_2023.2.sh

# Assign OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

init=step5_input
rest_prefix=step5_input
mini_prefix=step6.0_minimization
#equi_prefix=step6.%d_equilibration
prod_prefix=step7_production
prod_step=step7

# Minimization
# In the case that there is a problem during minimization using a single precision of GROMACS, please try to use 
# a double precision of GROMACS only for the minimization step.
mkdir em
cd em
gmx grompp -f ../mdp/${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ../${init}.gro -r ../${rest_prefix}.gro -p ../topol.top -n ../index.ndx -maxwarn 4
#gmx_d mdrun -v -deffnm ${mini_prefix}
gmx mdrun -s ${mini_prefix}.tpr -deffnm ${mini_prefix} -nb gpu -gpu_id 0 -ntmpi 1 
cd ..

## Equilibration
mkdir equil
cd equil
cnt=1
cntmax=6
cp ../em/${mini_prefix}.gro .
##while ( ${cnt} <= ${cntmax} )
while [ `echo "${cnt} <= ${cntmax}" |bc` -eq 1 ]
do
    pcnt=`echo "${cnt}-1"|bc`
    #set istep = `printf ${equi_prefix} ${cnt}`
    #set pstep = `printf ${equi_prefix} ${pcnt}`
    #if ( ${cnt} == 1 ) set pstep = ${mini_prefix}
    if [ `echo "${cnt} == 1" |bc` -eq 1 ];then pstep=${mini_prefix};else pstep=step6.${pcnt}_equilibration;fi

    gmx grompp -f ../mdp/step6.${cnt}_equilibration.mdp -o step6.${cnt}_equilibration.tpr -c ${pstep}.gro -r ${pstep}.gro -p ../topol.top -n ../index.ndx -maxwarn 4
    #gmx mdrun -v -deffnm ${istep}
    gmx mdrun -s step6.${cnt}_equilibration.tpr -deffnm step6.${cnt}_equilibration -nb gpu -bonded gpu -gpu_id 0 -pme gpu -ntmpi 1 
    cnt=`echo "${cnt}+1"|bc`
done
cd ..

# Production
mkdir prod
cd prod
cnt=1
cntmax=1
cp ../equil/step6.6_equilibration.gro .
#while ( ${cnt} <= ${cntmax} )
while [ `echo "${cnt} <= ${cntmax}" |bc` -eq 1 ]
do
    #@ pcnt = ${cnt} - 1
    pcnt=`echo "${cnt}-1"|bc`
    istep=${prod_step}_${cnt}
    pstep=${prod_step}_${pcnt}

    if [ `echo "${cnt} == 1" |bc` -eq 1 ];then
        pstep=step6.6_equilibration
        gmx grompp -f ../mdp/${prod_prefix}.mdp -o ${istep}.tpr -c ${pstep}.gro -r ${pstep}.gro -p ../topol.top -n ../index.ndx -maxwarn 4
    else
        gmx grompp -f ../mdp/${prod_prefix}.mdp -o ${istep}.tpr -c ${pstep}.gro -r ${pstep}.gro -t ${pstep}.cpt -p ../topol.top -n ../index.ndx -maxwarn 4
    fi
    #gmx mdrun -v -deffnm ${istep}
    gmx mdrun -s ${istep}.tpr -deffnm ${istep} -nb gpu -bonded gpu  -gpu_id 0 -pme gpu -ntmpi 1 
    cnt=`echo "${cnt}+1"|bc`
done
cd ..
