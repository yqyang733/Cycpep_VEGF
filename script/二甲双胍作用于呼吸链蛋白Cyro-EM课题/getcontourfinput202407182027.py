import os
from collections import defaultdict

class config:

    def __init__(self):

        self.files = os.path.join("fes.dat")

def submit(files):

    all_files = []
    with open(files) as f:
        f1 = f.readlines()
    for i in f1:
        all_files.append(i.replace("\n", ""))

    x_y_z = defaultdict(list)
    kkk = list()
    for i in all_files:
        with open(i) as f:
            f1 = f.readlines()
        for a in f1:
            if a.startswith("#") or a.startswith("\n"):
                pass
            else:
                # print(a)
                line = a.replace("\n", "").split()
                kkk.append((line[0], line[1]))
                x_y_z[(line[0], line[1])].append(float(line[2]))

    rt = open("contsurf_input.csv", "w")
    for i in kkk:
        if len(x_y_z[i]) == 1:
            rt.write(i[0]+","+i[1]+","+str(x_y_z[i][0])+"\n")
        else:
            val = sum(x_y_z[i])/len(x_y_z[i])
            rt.write(i[0]+","+i[1]+","+str(val)+"\n")

def main():

    settings = config()  
    submit(settings.files)

if __name__ == '__main__':
    main()
