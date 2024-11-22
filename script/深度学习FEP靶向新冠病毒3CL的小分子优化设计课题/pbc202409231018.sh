mkdir -p analysis_1/pbc
cd analysis_1/pbc
echo 24|gmx trjconv -f ../../npt/npt.gro -s ../../prod/prod.tpr -o new.pdb -n ../../index.ndx
echo q|gmx make_ndx -f new.pdb -o index_new.ndx
echo 0|gmx trjconv -f ../../prod_1/prod.xtc -s new.pdb -o md_pbcwhole_new.xtc -pbc whole -n index_new.ndx
echo 0|gmx trjconv -f md_pbcwhole_new.xtc -s new.pdb -o md_pbcnojump_new.xtc -pbc nojump -n index_new.ndx
echo 0 0|gmx trjconv -f md_pbcnojump_new.xtc -s new.pdb -o md_pbcmol_new.xtc -pbc mol -ur compact -center -n index_new.ndx
echo 0 0|gmx trjconv -f md_pbcmol_new.xtc -s new.pdb -o md_pbcfit_all_new.xtc -fit rot+trans -n index_new.ndx
rm md_pbcwhole_new.xtc md_pbcnojump_new.xtc md_pbcmol_new.xtc
cd ..
mkdir rmsd
cd rmsd
echo 0 0|gmx rms -f ../pbc/md_pbcfit_all_new.xtc -s ../pbc/new.pdb -o rms_whole.xvg -n ../pbc/index_new.ndx
echo 0 13|gmx rms -f ../pbc/md_pbcfit_all_new.xtc -s ../pbc/new.pdb -o rms_lig.xvg -n ../pbc/index_new.ndx
cd ..
mkdir contact
cd contact
cp ../../../../../contact.py .
python contact.py
cd ../../../
