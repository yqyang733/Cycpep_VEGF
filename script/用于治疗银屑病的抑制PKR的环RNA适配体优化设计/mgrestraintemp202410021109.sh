echo 1|gmx pdb2gmx -f RNA.pdb -o complex.gro -water tip3p -ignh
gmx editconf -f complex.gro -o newbox.gro -bt cubic -d 0.8
gmx solvate -cp newbox.gro -cs spc216.gro -p topol.top -o solv.gro
gmx grompp -f ~/file/gmx_file/ions.mdp -c solv.gro -p topol.top -o ions1.tpr -maxwarn 2
echo 3|gmx genion -s ions1.tpr -o solv_mg.gro -p topol.top -pname MG -nname CL -conc 0.015
gmx grompp -f ~/file/gmx_file/ions.mdp -c solv_mg.gro -p topol.top -o ions2.tpr -maxwarn 2
echo 5|gmx genion -s ions2.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -conc 0.15 -neutral
gmx make_ndx -f solv_ions.gro -o index.ndx