mkdir analysis;cd analysis
mkdir pbc;cd pbc

cp ../../index.ndx .
echo "[ atom ]" >> index.ndx
echo "1" >> index.ndx
source ~/../../software/profile.d/apps_gromacs_2023.2.sh
echo 17|gmx trjconv -f ../../prod/step4.1_equilibration.gro -s ../../prod/step5_1.tpr -o equil_1.pdb -n index.ndx
cp ../../../52I63I153I2DDD/analysis/pbc/equil_2.pdb .

export PATH="/public/home/yqyang/software/Miniconda3/envs/pymol/bin:$PATH"
cat > align.py << EOF
from pymol import cmd

cmd.load("equil_1.pdb")
cmd.load("equil_2.pdb")
cmd.align("equil_1 and polymer.protein", "equil_2 and polymer.protein")

cmd.set("retain_order", 1)
cmd.save("equil.pdb", "equil_1")
EOF
python align.py

gmx editconf -f equil.pdb -o equil.gro
tac equil.gro | sed '1s/.*/   10.00000   10.00000   10.00000/' | tac > equil.grotmp
mv equil.grotmp equil.gro
echo q|gmx make_ndx -f equil.gro -o index_1.ndx
cp -r ../../top* .
tac topol.top | sed '2,4d' | tac > topol.toptmp
mv topol.toptmp topol.top
gmx grompp -f ../../mdp/step4.0_minimization.mdp -o pbc.tpr -c equil.gro -r equil.gro -p topol.top -n index_1.ndx -maxwarn 4

cp ../../prod/step5_1.xtc prod.xtc
echo 19 0|gmx trjconv -f prod.xtc -s ../../prod/step5_1.tpr -o md_pbcmol_new.xtc -pbc atom -ur compact -center -n index.ndx  # 选 atom 和 system
echo 17|gmx trjconv -f md_pbcmol_new.xtc -s ../../prod/step5_1.tpr -o md_pbcwhole_new.xtc -pbc whole -n index.ndx  # 选 system
echo 1 0|gmx trjconv -f md_pbcwhole_new.xtc -s pbc.tpr -o md_pbcfit_all_new.xtc -fit rot+trans -n index_1.ndx 

rm md_pbcmol_new.xtc md_pbcwhole_new.xtc prod.xtc