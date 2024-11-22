import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plot():

    with open("zerror_0Ar.xvg_1") as f:
        f1 = f.readlines()
    
    x_1 = []
    y_1 = []
    y_lower_1 = []
    y_upper_1 = []

    for i in f1:
        if (i.startswith("#") or i.startswith("@")):
            pass
        else:
            line = i.replace("\n", "").split()
            x_1.append(float(line[0]))
            y_1.append(float(line[1]))
            y_lower_1.append(float(line[1])-float(line[2]))
            y_upper_1.append(float(line[1])+float(line[2]))

    with open("zerror_100Ar.xvg_1") as f:
        f1 = f.readlines()
    
    x_2 = []
    y_2 = []
    y_lower_2 = []
    y_upper_2 = []

    for i in f1:
        if (i.startswith("#") or i.startswith("@")):
            pass
        else:
            line = i.replace("\n", "").split()
            x_2.append(float(line[0]))
            y_2.append(float(line[1]))
            y_lower_2.append(float(line[1])-float(line[2]))
            y_upper_2.append(float(line[1])+float(line[2]))
    
    fig = plt.figure(figsize=(8,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()

    b_1, = plt.plot(x_1, y_1, linewidth=5, color='#66889E', label="0Ar")
    plt.fill_between(x_1, y_upper_1, y_lower_1, alpha=0.2, color='#66889E',)

    b_2, = plt.plot(x_2, y_2, linewidth=5, color='#C35C6A', label="100Ar")
    plt.fill_between(x_2, y_upper_2, y_lower_2, alpha=0.2, color='#C35C6A',)

    plt.xlabel('COM Distance/nm', fontproperties="Arial", fontsize=24, weight="bold")
    plt.ylabel('PMF/(Kcal/mol)', fontproperties="Arial", fontsize=24, weight="bold")
    plt.xticks(font="Arial", rotation=0, size=20, weight="bold")
    plt.yticks(font="Arial", size=20, weight="bold")

    plt.legend(handles=[b_1,b_2,],loc=(0.05,0.8),ncol=1,frameon=False,prop="Arial")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=18, weight="bold")

    # plt.xlim(-0.05, 2.00)
    # plt.ylim(-4, 8)

    plt.show()
    fig.savefig('pmf_curve_merge.pdf')

def main():

    # in_file = sys.argv[1]
    plot()

if __name__ == '__main__':
    main()