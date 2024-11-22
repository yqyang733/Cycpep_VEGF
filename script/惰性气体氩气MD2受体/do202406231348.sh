echo 0 17 0|gmx editconf -f step5_1.pdb -o newbox.gro -center 3 3 3 -box 12 6 6 -princ -n index.ndx  # 有时候需要沿着y轴或者z轴方向旋转180°。  -rotate 0 180 0
gmx solvate -cp newbox.gro -cs ~/file/gmx_file/spc216.gro -p topol.top -o solv.gro
gmx grompp -f ~/file/gmx_file/ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname SOD -nname CLA -neutral -conc 0.15
gmx make_ndx -f solv_ions.gro -o index.ndx