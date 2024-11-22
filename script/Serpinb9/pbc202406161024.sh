mkdir analysis
cd analysis/
mkdir pbc
cd pbc
cp ../../index.ndx .
echo "[ atom ]" >> index.ndx
echo "2725" >> index.ndx
echo 1|gmx trjconv -f ../../npt/npt.gro -s ../../prod/prod.tpr -o new.pdb -n index.ndx
echo 17 1|gmx trjconv -f ../../prod/prod.xtc -s ../../prod/prod.tpr -o md_pbcmol_new.xtc -pbc atom -ur compact -center -n index.ndx # 1 1
echo 1|gmx trjconv -f md_pbcmol_new.xtc -s ../../prod/prod.tpr -o md_pbcwhole_new.xtc -pbc whole -n index.ndx # 1
echo 1 1|gmx trjconv -f md_pbcwhole_new.xtc -s ../../prod/prod.tpr -o md_pbcfit_all_new.xtc -fit rot+trans -n index.ndx # 1
rm md_pbcmol_new.xtc md_pbcwhole_new.xtc
mkdir cluster
cd cluster
echo 1 1|gmx cluster -s ../../../prod/prod.tpr -f ../md_pbcfit_all_new.xtc -g -dist -sz -clid -cl -method linkage -cutoff 0.3 -n ../index.ndx
cd ../
mkdir rmsd
cd rmsd
echo 2 2| gmx rms -f ../md_pbcfit_all_new.xtc -s ../new.pdb -o rms.xvg
cd ..