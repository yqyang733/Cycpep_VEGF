echo 1|gmx pdb2gmx -f complex.pdb -o build.gro -water tip3p -ignh
gmx editconf -f build.gro -o newbox.gro -bt cubic -d 1.0
gmx grompp -f ~/file/gmx_file/ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo 13|gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15
gmx make_ndx -f solv_ions.gro -o index.ndx # SOLU SOLV