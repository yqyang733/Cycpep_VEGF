import os
from collections import defaultdict

class config:

    def __init__(self):

        self.f_in = os.path.join(".", "zrror_normal_c1.dat")  
        self.cut = float(-6.936843)    

def merge_pullf(f_in, cut):

    with open(f_in) as f:
        f1 = f.readlines()

    ref_y = float(f1[0].replace("\n", "").split(",")[1])-cut
    lst_all = []
    for i in f1:
        if i.startswith("#") or i.startswith("@"):
            pass
        else:
            line = i.replace("\n", "").split(",")
            lst_all.append(line[0]+","+str(float(line[1])-ref_y)+","+str(float(line[2])-ref_y)+","+str(float(line[3])-ref_y)+"\n")
    
    rt = open("zrror_normal_c1_1.dat", "w")
    rt.write("".join(lst_all))
    rt.close()

def main():

    settings = config()  
    merge_pullf(settings.f_in, settings.cut)  

if __name__ == '__main__':
    main()