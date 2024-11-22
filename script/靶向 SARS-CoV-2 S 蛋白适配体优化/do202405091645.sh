gmx trjcat -f ../../prod/prod.xtc ../../prod/prod.part0002.xtc ../../prod1/prod1.xtc -settime -o prod.xtc
mkdir analysis
cd analysis
mkdir pbc
cd pbc
cp ../../index.ndx .
echo "[ atom ]" >> index.ndx
echo "1236" >> index.ndx
echo 0|gmx trjconv -f ../../npt/npt.gro -s ../../prod/prod.tpr -o new.pdb -n index.ndx
echo 23 0|gmx trjconv -f ../../prod/prod.xtc -s ../../prod/prod.tpr -o md_pbcmol_new.xtc -pbc atom -ur compact -center -n index.ndx # 1 1
echo 0|gmx trjconv -f md_pbcmol_new.xtc -s ../../prod/prod.tpr -o md_pbcwhole_new.xtc -pbc whole -n index.ndx # 1
echo 21 0|gmx trjconv -f md_pbcwhole_new.xtc -s ../../prod/prod.tpr -o md_pbcfit_all_new.xtc -fit rot+trans -n index.ndx # 1
rm md_pbcmol_new.xtc md_pbcwhole_new.xtc

echo 21 21|gmx rms -s ../../prod/prod.tpr -f md_pbcfit_all_new.xtc -n index.ndx # 1 1
echo 0|gmx trjconv -s ../../prod/prod.tpr -f md_pbcfit_all_new.xtc -skip 100 -o md_pbcfit_all_new_1.xtc
rm md_pbcfit_all_new.xtc
mkdir cluster
cd cluster
echo -e "4|12\nq\n" |gmx make_ndx -f ../../../prod/prod.tpr -o index.ndx
echo 21 0|gmx cluster -s ../../../prod/prod.tpr -f ../md_pbcfit_all_new_1.xtc -g -dist -sz -clid -cl -method linkage -cutoff 0.3 -n index.ndx
