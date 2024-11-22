import sys
import numpy as np
import matplotlib.pyplot as plt

def plot(in_file):

    with open(in_file) as f:
        f1 = f.readlines()
    
    x = []
    y = []
    y_lower = []
    y_upper = []

    for i in f1:
        if (i.startswith("#") or i.startswith("@")):
            pass
        else:
            line = i.replace("\n", "").split()
            x.append(float(line[0]))
            y.append(float(line[1]))
            y_lower.append(float(line[1])-float(line[2]))
            y_upper.append(float(line[1])+float(line[2]))
    
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()

    b1, = plt.plot(x, y, linewidth=3.5,)
    plt.fill_between(x, y_upper, y_lower, alpha=0.5,)

    plt.xlabel('Distance/nm', fontproperties="Times New Roman", fontsize=28, weight="bold")
    plt.ylabel('PMF/(Kcal/mol)', fontproperties="Times New Roman", fontsize=28, weight="bold")
    plt.xticks(font="Times New Roman", rotation=0, size=20)
    plt.yticks(font="Times New Roman", size=20)

    plt.show()
    fig.savefig('pmf_curve.pdf')

def main():

    in_file = sys.argv[1]
    plot(in_file)
    # 使用方法：python do.py zerror.xvg

if __name__ == '__main__':
    main()