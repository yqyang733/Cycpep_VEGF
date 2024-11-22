import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_1():

    with open("density_rt_1.csv") as f:
        f1 = f.readlines()
    x_rt_1 = []
    y_rt_1 = []
    for i in f1:
        line = i.strip().split(",")
        x_rt_1.append(float(line[0]))
        y_rt_1.append(float(line[1]))

    with open("density_rt_2.csv") as f:
        f1 = f.readlines()
    x_rt_2 = []
    y_rt_2 = []
    for i in f1:
        line = i.strip().split(",")
        x_rt_2.append(float(line[0]))
        y_rt_2.append(float(line[1]))

    with open("density_rt_3.csv") as f:
        f1 = f.readlines()
    x_rt_3 = []
    y_rt_3 = []
    for i in f1:
        line = i.strip().split(",")
        x_rt_3.append(float(line[0]))
        y_rt_3.append(float(line[1]))
    
    fig = plt.figure(figsize=(8,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax = plt.gca()

    b1, = plt.plot(x_rt_1, y_rt_1, linewidth=3.5, color='#66889E', label="dist1")
    # plt.fill_between(x_1, y_upper_1, y_lower_1, alpha=0.2, color='#66889E',)
    b2, = plt.plot(x_rt_2, y_rt_2, linewidth=3.5, color='#C35C6A', label="dist2")
    # plt.fill_between(x_2, y_upper_2, y_lower_2, alpha=0.2, color='#C35C6A',)
    b3, = plt.plot(x_rt_3, y_rt_3, linewidth=3.5, color='#7E527F', label="dist3")

    plt.xlabel('Distance/nm', fontproperties="Arial", fontsize=28, weight="bold")
    plt.ylabel('Probability Distribution', fontproperties="Arial", fontsize=28, weight="bold")
    plt.xticks(font="Arial", rotation=0, size=20, weight="bold")
    plt.yticks(font="Arial", size=20, weight="bold")

    plt.legend(handles=[b1,b2,b3],loc=(0.8,0.75),ncol=1,frameon=False,prop="Arial")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=18, weight="bold")

    plt.show()
    fig.savefig('probalility_fig.pdf')

def plot_2():

    with open("density_dg_1.csv") as f:
        f1 = f.readlines()
    x_dg_1 = []
    y_dg_1 = []
    for i in f1:
        line = i.strip().split(",")
        x_dg_1.append(float(line[0]))
        y_dg_1.append(float(line[1]))

    with open("density_dg_2.csv") as f:
        f1 = f.readlines()
    x_dg_2 = []
    y_dg_2 = []
    for i in f1:
        line = i.strip().split(",")
        x_dg_2.append(float(line[0]))
        y_dg_2.append(float(line[1]))

    with open("density_dg_3.csv") as f:
        f1 = f.readlines()
    x_dg_3 = []
    y_dg_3 = []
    for i in f1:
        line = i.strip().split(",")
        x_dg_3.append(float(line[0]))
        y_dg_3.append(float(line[1]))

    fig = plt.figure(figsize=(8,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax = plt.gca()

    b1, = plt.plot(x_dg_1, y_dg_1, linewidth=3.5, color='#66889E', label="dist1")
    # plt.fill_between(x_1, y_upper_1, y_lower_1, alpha=0.2, color='#66889E',)
    b2, = plt.plot(x_dg_2, y_dg_2, linewidth=3.5, color='#C35C6A', label="dist2")
    # plt.fill_between(x_2, y_upper_2, y_lower_2, alpha=0.2, color='#C35C6A',)
    b3, = plt.plot(x_dg_3, y_dg_3, linewidth=3.5, color='#7E527F', label="dist3")

    plt.xlabel('Distance/nm', fontproperties="Arial", fontsize=28, weight="bold")
    plt.ylabel('ΔG/(Kcal/mol)', fontproperties="Arial", fontsize=28, weight="bold")
    plt.xticks(font="Arial", rotation=0, size=20, weight="bold")
    plt.yticks(font="Arial", size=20, weight="bold")

    plt.legend(handles=[b1,b2,b3],loc=(0.8,0.75),ncol=1,frameon=False,prop="Arial")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=18, weight="bold")

    plt.show()
    fig.savefig('ddg_fig.pdf')

def main():

    plot_1()
    plot_2()

if __name__ == '__main__':
    main()