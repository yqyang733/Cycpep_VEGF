with open("dist.xvg") as f:
    f1 = f.readlines()
f2 = []
for i in f1:
    if i.startswith("#") or i.startswith("@"):
        pass
    else:
        f2.append(i)

with open("dists_min.csv") as f:
    f1 = f.readlines()
dist_lst = []
for i in f1:
    line = i.strip().split(",")
    dist_lst.append(float(line[1]))

cut = 100000
for i in f2:
    line = i.strip().split()
    dis_tmp = (float(line[1])-dist_lst[0])**2 + (float(line[2])-dist_lst[1])**2 + (float(line[3])-dist_lst[2])**2
    if dis_tmp < cut:
        pick_t = i
        cut = dis_tmp

rt = open("pick_time.csv", "w")
rt.write(pick_t)
rt.close()