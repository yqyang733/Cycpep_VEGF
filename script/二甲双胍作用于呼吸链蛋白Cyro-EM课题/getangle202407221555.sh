cp ../../../index.ndx .
# 在index.ndx中加入以下计算angle的数据：
# [ angle ]
# 14783 14799 14789
gmx angle -f ../../../prod/step7_1.xtc -n index.ndx -ov entrance.xvg