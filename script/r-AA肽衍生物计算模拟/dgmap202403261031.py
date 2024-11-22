import numpy as np
import math

with open("distave.xvg") as f:
    f1 = f.readlines()
f2 = []
for i in f1:
    if i.startswith("#") or i.startswith("@"):
        pass
    else:
        f2.append(i)
f1 = f2
dis_1 = list()
dis_2 = list()
for i in f1:
    line = i.strip().split()
    dis_1.append(float(line[1]))
    dis_2.append(float(line[2]))

dis1_start = min(dis_1)
dis1_end = max(dis_1)
dis2_start = min(dis_2)
dis2_end = max(dis_2)
margin = 20
margin1_value = (dis1_end - dis1_start)/margin
margin2_value = (dis2_end - dis2_start)/margin
matrix_xy = np.zeros((margin, margin))

for i in f1:
    line = i.strip().split()
    print(line)
    print(dis2_start)
    print(dis1_start)
    print(margin2_value)
    print((float(line[2])-dis2_start)//margin2_value)
    matrix_xy[int((float(line[2])-dis2_start-0.000000000000001)//margin2_value)][int((float(line[1])-dis1_start-0.000000000000001)//margin1_value)] += 1

matrix_xy_density = matrix_xy/len(f1)
min_v = 0.000001

rt_xy = open("density_rt.csv", "w")
rt_dg = open("density_dg.csv", "w")
for aa in range(margin):
    y_tmp = dis2_start + margin2_value/2 + margin2_value*aa
    for bb in range(margin):
        x_tmp = dis1_start + margin1_value/2 + margin1_value*bb
        rt_xy.write(str(x_tmp)+","+str(y_tmp)+","+str(matrix_xy_density[aa][bb])+"\n")
        if matrix_xy_density[aa][bb] == 0:
            rt_dg.write(str(x_tmp)+","+str(y_tmp)+","+str(-0.596*math.log(min_v))+"\n")
        else:
            rt_dg.write(str(x_tmp)+","+str(y_tmp)+","+str(-0.596*math.log(matrix_xy_density[aa][bb]))+"\n")
rt_xy.close()
rt_dg.close()

with open("density_dg.csv") as f:
    f1 = f.readlines()
rt = open("dg.csv", "w")
all_dg = []
for i in f1:
    line = i.strip().split(",")
    all_dg.append(float(line[2]))
min_dg = min(all_dg)
for i in f1:
    line = i.strip().split(",")
    rt.write(line[0]+","+line[1]+","+str(float(line[2])-min_dg)+"\n")
rt.close()

min_max = open("min_max.txt", "w")
min_max.write("dis_1最小值为"+str(min(dis_1))+";dis_1最大值为"+str(max(dis_1))+"\n")  
min_max.write("dis_2最小值为"+str(min(dis_2))+";dis_2最大值为"+str(max(dis_2))+"\n")  
