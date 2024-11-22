import sys
import numpy as np
from collections import defaultdict

def plot(in_file, wins):

    wins_name = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        wins_name.append(i.replace("\n", ""))

    with open(in_file) as f:
        f1 = f.readlines()
    
    wins_data = defaultdict(list)

    for i in f1:
        if (i.startswith("#") or i.startswith("@")):
            pass
        else:
            line = i.replace("\n", "").split()
            for j in range(len(wins_name)):
                wins_data[wins_name[j]].append([float(line[0]),float(line[j+1])])
    
    rt = open("hist_plotly_in.csv", "w")
    rt.write("windows,x,y\n")
    for key in wins_data:
        for i in range(len(wins_data[key])):
            rt.write(str(key)+","+str(wins_data[key][i][0])+","+str(wins_data[key][i][1])+"\n")
    rt.close()

def main():

    in_file = sys.argv[1]
    wins = sys.argv[2]
    plot(in_file, wins)

if __name__ == '__main__':
    main()