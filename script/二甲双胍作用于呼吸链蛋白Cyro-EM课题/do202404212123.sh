gmx trjconv -f ../../frame.gro -s ../../pull/pull.tpr -o equil.pdb -n ../../index.ndx 
# 选择SOLU
gmx trjconv -f ../../pull/pull.xtc -s ../../pull/pull.tpr -o pull_pbc.xtc -pbc mol -ur compact -center -n ../../index.ndx
# 选择SOLU_MEMB 和 SOLU