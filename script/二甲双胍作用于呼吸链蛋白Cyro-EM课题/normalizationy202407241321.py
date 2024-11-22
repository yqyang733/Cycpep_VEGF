with open("zerror_3all.dat") as f:
    f1 = f.readlines()

zero_base = float(f1[0].split(",")[1])
rt = open("zerror_3all_normal.dat", "w")
for i in f1:
    line = i.split(",")
    rt.write(str(line[0])+","+str(float(line[1])-zero_base)+","+str(float(line[1])-zero_base-float(line[2]))+","+str(float(line[1])-zero_base+float(line[2]))+"\n")
rt.close()