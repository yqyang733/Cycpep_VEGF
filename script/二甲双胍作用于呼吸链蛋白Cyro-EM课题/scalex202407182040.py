with open("contsurf_input.csv") as f:
    f1 = f.readlines()

rt = open("contsurf_input_1.csv", "w")

for i in f1:
    line = i.split(",")
    if float(line[0]) <= 2:
        rt.write(i)
    else:
        rt.write(str(float(line[0])*1.1)+","+line[1]+","+line[2])

rt.close()