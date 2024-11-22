import os
from collections import defaultdict

class config:

    def __init__(self):

        self.win_name = os.path.join(".", "wins.dat")  
        self.xu_number = int(10)  

def merge_pullf(win_name, xu_number):

    win_n = []
    with open(win_name) as f:
        f1 = f.readlines()
    for i in f1:
        win_n.append(i.replace("\n", ""))

    for i in win_n:

        multidict_time_f = defaultdict(list)  
        head = []
        with open(os.path.join(".", i, "prod", "prodf.xvg")) as f:
            f1 = f.readlines()

        for j in f1:
            if j.startswith("#") or j.startswith("@"):
                head.append(j)
            else:
                line = j.replace("\n", "").split()
                if len(line) == 2:
                    multidict_time_f[line[0]].append(line[1])

        for a in range(2, xu_number):
            if os.path.exists(os.path.join(".", i, "prod", "prodf.part000{:d}.xvg").format(a)):
                with open(os.path.join(".", i, "prod", "prodf.part000{:d}.xvg").format(a)) as f:
                    f1 = f.readlines()
                for j in f1:
                    if j.startswith("#") or j.startswith("@"):
                        pass
                    else:
                        line = j.replace("\n", "").split()
                        if len(line) == 2:
                            multidict_time_f[line[0]].append(line[1])

        rt = open(os.path.join(".", i, "prod", "prodf_all.xvg"), "w")
        rt.write("".join(head))
        for b in multidict_time_f:
            rt.write(b + "    " + multidict_time_f[b][0] + "\n")

def main():

    settings = config()  
    merge_pullf(settings.win_name, settings.xu_number)  

if __name__ == '__main__':
    main()