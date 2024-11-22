import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def process(in_file):
    with open(in_file) as f:
        f1 = f.readlines()
    
    x = []
    y = []
    z = []
    x_1 = []
    y_1 = []
    z_1 = []

    for i in range(len(f1)):
        line = f1[i].replace("\n", "").split(",")
        if (i+1)%20 == 0:
            x_1.append(float(line[0]))
            y_1.append(float(line[1]))
            z_1.append(float(line[2]))
            x.append(x_1)
            y.append(y_1)
            z.append(z_1)
            x_1 = []
            y_1 = []
            z_1 = []
        else:
            x_1.append(float(line[0]))
            y_1.append(float(line[1]))
            z_1.append(float(line[2]))

    fig,ax = plt.subplots(figsize=(12,10))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.97, bottom=0.32) 
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    # plt.ylim(-0.1, 10.1)
    plt.xlim(0.385, 0.43)
    cs = ax.contourf(x, y, z, 50, levels=np.linspace(0,6.45,100), cmap = plt.cm.jet) # vmax和levels需要按情况进行适当修改。  vmin=-0.03, vmax = 0.35,
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('ΔG/(Kcal/mol)',fontdict={'family':'Arial', 'weight':'bold', 'size':30,})
    cbar.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ]) # 需要按情况根据最大值进行适当调整。 

    plt.xlabel("Distance/nm",font = "Arial",fontsize=36,weight="bold") # 根据情况修改
    plt.ylabel("Distance/nm",font = "Arial",fontsize=36,weight="bold") # 根据情况修改
    plt.xticks(font="Arial",size=30,weight="bold",rotation=45)
    plt.yticks(font="Arial",size=30,weight="bold")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    plt.show()
    fig.savefig("Figure_xy.pdf") # 根据情况修改

def main():

    in_file = "dg.csv" # 根据情况修改
    process(in_file)

if __name__ == '__main__':
    main()