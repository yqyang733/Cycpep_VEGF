import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt

def plot(in_file):

    with open(in_file) as f:
        f1 = f.readlines()
    
    x_1 = []
    y_1 = []
    y_lower_1 = []
    y_upper_1 = []

    for i in f1:
        line = i.replace("\n", "").split(",")
        x_1.append(float(line[0]))
        y_1.append(float(line[1]))
        y_lower_1.append(float(line[2]))
        y_upper_1.append(float(line[3]))

    fig = plt.figure(figsize=(10,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()

    b1, = plt.plot(x_1, scipy.signal.savgol_filter(y_1, 20, 10), linewidth=5, color='#C35C6A', label="open(class1)")
    plt.fill_between(x_1, scipy.signal.savgol_filter(y_upper_1, 20, 10), scipy.signal.savgol_filter(y_lower_1, 20, 10), alpha=0.1, color='#C35C6A',)
    # b1, = plt.plot(x_1, y_1, linewidth=5, color='#66889E', label="open(class1)")
    # plt.fill_between(x_1, y_upper_1, y_lower_1, alpha=0.1, color='#66889E',)
    # b2, = plt.plot(x_2, scipy.signal.savgol_filter(y_2, 20, 3), linewidth=5, color='#7E527F', label="open(class3)")
    # plt.fill_between(x_2, scipy.signal.savgol_filter(y_upper_2, 20, 3), scipy.signal.savgol_filter(y_lower_2, 20, 3), alpha=0.1, color='#7E527F',)
    # # b2, = plt.plot(x_2, y_2, linewidth=3.5, color='#66889E', label="open(class3)")
    # # plt.fill_between(x_2, y_upper_2, y_lower_2, alpha=0.2, color='#66889E',)
    # b3, = plt.plot(x_3, scipy.signal.savgol_filter(y_3, 20, 3), linewidth=5, color='#C35C6A', label="close(class5)")
    # plt.fill_between(x_3, scipy.signal.savgol_filter(y_upper_3, 20, 3), scipy.signal.savgol_filter(y_lower_3, 20, 3), alpha=0.1, color='#C35C6A',)
    # b3, = plt.plot(x_3, y_3, linewidth=3.5, color='#66889E', label="open(class5)")
    # plt.fill_between(x_3, y_upper_3, y_lower_3, alpha=0.2, color='#66889E',)

    plt.xlabel('COM Distance (nm)', fontproperties="Arial", fontsize=24, weight="bold")
    plt.ylabel('PMF (kcal/mol)', fontproperties="Arial", fontsize=24, weight="bold")
    plt.xticks(font="Arial", rotation=0, size=20, weight="bold")
    plt.yticks(font="Arial", size=20, weight="bold")

    # plt.legend(handles=[b1,],loc=(0.1,0.14),ncol=1,frameon=False,prop="Arial")
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=24, weight="bold")

    plt.ylim(-18, 10)

    plt.show()
    # fig.savefig('pmf_c1c4c5.pdf')
    fig.savefig('pmf_.pdf')

def main():

    in_file = sys.argv[1]
    plot(in_file)

if __name__ == '__main__':
    main()