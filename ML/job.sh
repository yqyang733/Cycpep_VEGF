#!/bin/bash

#PBS -N aa
#PBS -j oe
#PBS -q sugon_new
#PBS -l nodes=node40:ppn=28
#PBS -l walltime=2400:00:00

export PATH=/public/home/yqyang/software/Miniconda3/envs/Cycpep_VEGF/bin:$PATH

ulimit -s unlimited
ulimit -l unlimited

cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE | wc -l`
echo "Starting run at" `date`

python main.py

echo "Finished run at" `date`

