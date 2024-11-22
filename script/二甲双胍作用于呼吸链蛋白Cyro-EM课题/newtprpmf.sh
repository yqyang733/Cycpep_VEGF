mkdir analysis
cd analysis
mkdir run2
cd run2
for i in `seq 1 30`;do echo win_${i} >> submit.dat;done
cp ../../index.ndx . # 将新的用于参考的坐标的原子编号写进index.ndx中重新生成新的tpr文件。
for i in `cat submit.dat`;do gmx grompp -f ../../mdp/prod.mdp -c ../../${i}/${i}.gro -p ../../topol.top -o pmf${i}.tpr -r ../../${i}/${i}.gro -maxwarn 4 -n ../../index.ndx;done
for i in `cat submit.dat`;do cp ../../${i}/prod/prodf_all.xvg ./pmf${i}_pullf.xvg;done
