for i in `cat lst`;do cat ${i} >> HILLS;done
source ~/../../software/profile.d/apps_gromacs_2022.5_mpi_plumed.sh
plumed sum_hills --hills HILLS --outfile fes_1.dat --min 0,-pi --max 5,pi --bin 40,30 --mintozero