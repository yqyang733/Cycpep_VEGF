from pymol import cmd
import numpy as np

margin = 100
start_ar = 639
end_ar = 838

matrix_xz = np.zeros((margin, margin))
matrix_xy = np.zeros((margin, margin))
matrix_yz = np.zeros((margin, margin))

x_more = []
y_more = []
z_more = []
all_cords = []

cmd.load("../equil.pdb")
cmd.load_traj("../md_pbcfit_all_new.xtc")
num_states = cmd.count_states()
for i in range(num_states):
    for a in range(start_ar, end_ar+1):
        tmp_coor = cmd.centerofmass("resi %d"%(a), state=i+1)
        if tmp_coor[0] > 0 and tmp_coor[0] < 100 and tmp_coor[1] > 0 and tmp_coor[1] < 100 and tmp_coor[2] > 0 and tmp_coor[2] < 100:
            all_cords.append(tmp_coor)

for i in all_cords:
    matrix_xz[int(np.floor(i[2]))][int(np.floor(i[0]))] += 1
    matrix_xy[int(np.floor(i[1]))][int(np.floor(i[0]))] += 1
    matrix_yz[int(np.floor(i[2]))][int(np.floor(i[1]))] += 1

matrix_xz_density = matrix_xz/num_states
matrix_xy_density = matrix_xy/num_states
matrix_yz_density = matrix_yz/num_states

rt_xz = open("xz_density.csv", "w")
for aa in range(margin):
    y_tmp = aa + 0.5
    for bb in range(margin):
        x_tmp = bb + 0.5
        rt_xz.write(str(x_tmp)+","+str(y_tmp)+","+str(matrix_xz_density[aa][bb])+"\n")
rt_xz.close()

rt_xy = open("xy_density.csv", "w")
for aa in range(margin):
    y_tmp = aa + 0.5
    for bb in range(margin):
        x_tmp = bb + 0.5
        rt_xy.write(str(x_tmp)+","+str(y_tmp)+","+str(matrix_xy_density[aa][bb])+"\n")
rt_xy.close()

rt_yz = open("yz_density.csv", "w")
for aa in range(margin):
    y_tmp = aa + 0.5
    for bb in range(margin):
        x_tmp = bb + 0.5
        rt_yz.write(str(x_tmp)+","+str(y_tmp)+","+str(matrix_yz_density[aa][bb])+"\n")
rt_yz.close()

min_max = open("min_max.txt", "w")
min_max.write("xy平面最小值为"+str(matrix_xy_density.min())+";xy平面最大值为"+str(matrix_xy_density.max())+"\n")
min_max.write("xz平面最小值为"+str(matrix_xz_density.min())+";xz平面最大值为"+str(matrix_xz_density.max())+"\n")
min_max.write("yz平面最小值为"+str(matrix_yz_density.min())+";yz平面最大值为"+str(matrix_yz_density.max())+"\n")    
