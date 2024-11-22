
for i in range(100):
    i = str(i).zfill(2)
    rt = open("job_topol_"+str(i)+".sh", "w")
    rt.write('''#!/bin/bash

#PBS -N radiotherapy
#PBS -j oe
#PBS -q sugon_new
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2400:00:00

export PATH=/public/home/yqyang/software/Miniconda3/envs/pytorch/bin:$PATH

ulimit -s unlimited
ulimit -l unlimited

cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE | wc -l`
echo "Starting run at" `date`

python do202405091020.py ZINC_{0} com11-top-similarity-{0}

echo "Finished run at" `date`

'''.format(i))
    rt.close()