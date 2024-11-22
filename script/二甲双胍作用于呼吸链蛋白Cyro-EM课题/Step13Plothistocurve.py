import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot(in_file, wins):

    wins_name = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        wins_name.append(i.replace("\n", ""))

    with open(in_file) as f:
        f1 = f.readlines()
    
    x = []
    wins_data = defaultdict(list)

    for i in f1:
        if (i.startswith("#") or i.startswith("@")):
            pass
        else:
            line = i.replace("\n", "").split()
            x.append(float(line[0]))
            for j in range(len(wins_name)):
                wins_data[j].append(float(line[j+1]))
    
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()

    for i in range(len(wins_name)):
        plt.plot(x, wins_data[i], linewidth=2, label=wins_name[i])

    plt.xlabel('Distance/nm', fontproperties="Times New Roman", fontsize=28, weight="bold")
    plt.ylabel('Count', fontproperties="Times New Roman", fontsize=28, weight="bold")
    plt.xticks(font="Times New Roman", rotation=0, size=20)
    plt.yticks(font="Times New Roman", size=20)

    plt.legend(loc=(1.01, 0),)

    plt.show()
    fig.savefig('Histo_curve.pdf')


def main():

    in_file = sys.argv[1]
    wins = sys.argv[2]
    plot(in_file, wins)

if __name__ == '__main__':
    main()