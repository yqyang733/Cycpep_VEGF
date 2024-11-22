with open("density_dg_1.csv") as f:
    f1 = f.readlines()
dict_1 = dict()
for i in f1:
    line = i.strip().split(",")
    dict_1[float(line[0])] = float(line[1])
sorted_dict_1 = dict(sorted(dict_1.items(), key=lambda item: item[1]))

with open("density_dg_2.csv") as f:
    f1 = f.readlines()
dict_2 = dict()
for i in f1:
    line = i.strip().split(",")
    dict_2[float(line[0])] = float(line[1])
sorted_dict_2 = dict(sorted(dict_2.items(), key=lambda item: item[1]))

with open("density_dg_3.csv") as f:
    f1 = f.readlines()
dict_3 = dict()
for i in f1:
    line = i.strip().split(",")
    dict_3[float(line[0])] = float(line[1])
sorted_dict_3 = dict(sorted(dict_3.items(), key=lambda item: item[1]))

rt = open("dists_min.csv", "w")
rt.write("dist1,"+str(list(sorted_dict_1.keys())[0])+"\n"+"dist2,"+str(list(sorted_dict_2.keys())[0])+"\n"+"dist3,"+str(list(sorted_dict_3.keys())[0])+"\n")
rt.close()