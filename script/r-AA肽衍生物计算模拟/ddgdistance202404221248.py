import numpy as np
import math

with open("dist.xvg") as f:
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
dis_3 = list()
for i in f1:
    line = i.strip().split()
    dis_1.append(float(line[1]))
    dis_2.append(float(line[2]))
    dis_3.append(float(line[3]))

dis1_start = min(dis_1)
dis1_end = max(dis_1)
dis2_start = min(dis_2)
dis2_end = max(dis_2)
dis3_start = min(dis_3)
dis3_end = max(dis_3)
margin = 50
margin1_value = (dis1_end - dis1_start)/margin
margin2_value = (dis2_end - dis2_start)/margin
margin3_value = (dis3_end - dis3_start)/margin
# matrix_xy = np.zeros((margin, margin))
array_1 = np.zeros(margin)
array_2 = np.zeros(margin)
array_3 = np.zeros(margin)

for i in f1:
    line = i.strip().split()
    # print(line)
    # print(dis2_start)
    # print(dis1_start)
    # print(margin2_value)
    # print((float(line[2])-dis2_start)//margin2_value)
    # matrix_xy[int((float(line[2])-dis2_start-0.000000000000001)//margin2_value)][int((float(line[1])-dis1_start-0.000000000000001)//margin1_value)] += 1
    array_1[int((float(line[1])-dis1_start-0.000000000000001)//margin1_value)] += 1
    array_2[int((float(line[2])-dis2_start-0.000000000000001)//margin2_value)] += 1
    array_3[int((float(line[3])-dis3_start-0.000000000000001)//margin3_value)] += 1

array_1_density = array_1/len(f1)/margin1_value
array_2_density = array_2/len(f1)/margin2_value
array_3_density = array_3/len(f1)/margin3_value
min_v = 0.000001

rt_array_1 = open("density_rt_1.csv", "w")
rt_array_1_dg = open("density_dg_1.csv", "w")
for aa in range(margin):
    x_tmp = dis1_start + margin1_value/2 + margin1_value*aa
    rt_array_1.write(str(x_tmp)+","+str(array_1_density[aa])+"\n")
    if array_1_density[aa] == 0:
        rt_array_1_dg.write(str(x_tmp)+","+str(-0.596*math.log(min_v))+"\n")
    else:
        rt_array_1_dg.write(str(x_tmp)+","+str(-0.596*math.log(array_1_density[aa]))+"\n")
rt_array_1.close()
rt_array_1_dg.close()

# with open("density_dg.csv") as f:
#     f1 = f.readlines()
# rt = open("dg.csv", "w")
# all_dg = []
# for i in f1:
#     line = i.strip().split(",")
#     all_dg.append(float(line[2]))
# min_dg = min(all_dg)
# for i in f1:
#     line = i.strip().split(",")
#     rt.write(line[0]+","+line[1]+","+str(float(line[2])-min_dg)+"\n")
# rt.close()

# min_max = open("min_max.txt", "w")
# min_max.write("dis_1最小值为"+str(min(dis_1))+";dis_1最大值为"+str(max(dis_1))+"\n")  
# min_max.write("dis_2最小值为"+str(min(dis_2))+";dis_2最大值为"+str(max(dis_2))+"\n")

rt_array_2 = open("density_rt_2.csv", "w")
rt_array_2_dg = open("density_dg_2.csv", "w")
for aa in range(margin):
    x_tmp = dis2_start + margin2_value/2 + margin2_value*aa
    rt_array_2.write(str(x_tmp)+","+str(array_2_density[aa])+"\n")
    if array_2_density[aa] == 0:
        rt_array_2_dg.write(str(x_tmp)+","+str(-0.596*math.log(min_v))+"\n")
    else:
        rt_array_2_dg.write(str(x_tmp)+","+str(-0.596*math.log(array_2_density[aa]))+"\n")
rt_array_2.close()
rt_array_2_dg.close()

rt_array_3 = open("density_rt_3.csv", "w")
rt_array_3_dg = open("density_dg_3.csv", "w")
for aa in range(margin):
    x_tmp = dis3_start + margin3_value/2 + margin3_value*aa
    rt_array_3.write(str(x_tmp)+","+str(array_3_density[aa])+"\n")
    if array_3_density[aa] == 0:
        rt_array_3_dg.write(str(x_tmp)+","+str(-0.596*math.log(min_v))+"\n")
    else:
        rt_array_3_dg.write(str(x_tmp)+","+str(-0.596*math.log(array_3_density[aa]))+"\n")
rt_array_3.close()
rt_array_3_dg.close()
