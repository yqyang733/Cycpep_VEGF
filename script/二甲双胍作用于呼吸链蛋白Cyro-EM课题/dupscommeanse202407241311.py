import numpy as np

dup1 = []
with open("zerror_all_dup1.xvg") as f:
    f1 = f.readlines()
for i in f1:
    line = i.split()
    dup1.append((float(line[0]),float(line[1])))

dup2 = []
with open("zerror_all_dup2.xvg") as f:
    f1 = f.readlines()
for i in f1:
    line = i.split()
    dup2.append((float(line[0]),float(line[1])))

dup3 = []
with open("zerror_all_dup3.xvg") as f:
    f1 = f.readlines()
for i in f1:
    line = i.split()
    dup3.append((float(line[0]),float(line[1])))

flag_1 = 10
flag_2 = 10 
dups_final = []
for i in dup1:
    print(i[0])
    if i[0] >= -3.56 and i[0] <= 3.45:
        for a in dup2:
            if abs(a[0]-i[0]) < flag_1:
                flag_1 = abs(a[0]-i[0])
                final_dup2 = a
                print(final_dup2)
        for b in dup3:
            if abs(b[0]-i[0]) < flag_2:
                flag_2 = abs(b[0]-i[0])
                final_dup3 = b
                print(final_dup3)
        dups_final.append(((i[0]+final_dup2[0]+final_dup3[0])/3,(i[1]+final_dup2[1]+final_dup3[1])/3,np.std([i[1],final_dup2[1],final_dup3[1]])/np.sqrt(3)))
    flag_1 = 10
    flag_2 = 10 

rt = open("zerror_3all.dat", "w")
for i in dups_final:
    rt.write(str(i[0])+","+str(i[1])+","+str(i[2])+"\n")
rt.close()