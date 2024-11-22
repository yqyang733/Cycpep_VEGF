SMD_path=/public/home/yqyang/biguanides_respiratory/MF8/close/MF8_2/PMF/SMD_2
cp ${SMD_path}/topol.top .
cp ${SMD_path}/index.ndx .
a=1;for i in `cat time_pick.dat`;do mkdir win_${a};echo 0|gmx trjconv -s ${SMD_path}/pull/pull.tpr -f ${SMD_path}/pull/prod_all.xtc -o ./win_${a}/win_${a}.gro -b ${i} -e ${i};a=`echo "${a}+1"|bc`;done