#!/bin/bash

#PBS -N BRICS
#PBS -j oe
#PBS -q sugon_fep
#PBS -l nodes=1:ppn=128
#PBS -l walltime=2400:00:00

export PATH=/public/home/yqyang/software/Anaconda3-2022.05/envs/pymol/bin:$PATH

ulimit -s unlimited
ulimit -l unlimited

cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE | wc -l`
echo "Starting similarity run at" `date`

python utility.py

echo "Finished similarity run at" `date`
