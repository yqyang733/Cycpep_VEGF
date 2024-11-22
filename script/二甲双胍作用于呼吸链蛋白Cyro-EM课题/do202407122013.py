import os
from collections import defaultdict

class config:

    def __init__(self):

        self.f_in = os.path.join(".", "zerror.xvg")  
        self.cut = float(0)    

def merge_pullf(f_in, cut):

    with open(f_in) as f:
        f1 = f.readlines()

    ref_y = float(f1[-1].replace("\n", "").split("\t")[1])
    lst_all = []
    for i in f1:
        if i.startswith("#") or i.startswith("@"):
            pass
        else:
            line = i.replace("\n", "").split("\t")
            if float(line[0]) >= cut:
                lst_all.append(str(cut-float(line[0]))+","+str(float(line[1])-ref_y)+","+str(float(line[1])-ref_y-float(line[2]))+","+str(float(line[1])-ref_y+float(line[2]))+"\n")
    
    rt = open("zrror_input.dat", "w")
    rt.write("".join(lst_all[::-1]))
    rt.close()

def main():

    settings = config()  
    merge_pullf(settings.f_in, settings.cut)  

if __name__ == '__main__':
    main()