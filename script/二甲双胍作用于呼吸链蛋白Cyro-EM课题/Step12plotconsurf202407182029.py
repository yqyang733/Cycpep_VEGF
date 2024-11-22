import sys
import numpy as np
import matplotlib.pyplot as plt

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
        if (i+1)%41 == 0:
            x_1.append(float(line[0]))
            y_1.append(180 - float(line[1])*180/3.1415926)
            if float(line[2])/4.184 >= 20:
                z_1.append(20)
            else:
                z_1.append(float(line[2])/4.184)
            x.append(x_1)
            y.append(y_1)
            z.append(z_1)
            x_1 = []
            y_1 = []
            z_1 = []
        else:
            x_1.append(float(line[0]))
            y_1.append(180 - float(line[1])*180/3.1415926)
            if float(line[2])/4.184 >= 20:
                z_1.append(20)
            else:
                z_1.append(float(line[2])/4.184)

    fig,ax = plt.subplots(figsize=(12,10))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2) 
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    plt.ylim(0, 70)
    plt.xlim(-1, 2.5)
    cs = ax.contourf(x, y, z, 100, vmax = 20, cmap=plt.cm.jet)
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Free energy/(kcal/mol)',fontdict={'family':'Arial', 'weight':'bold', 'size':36,})
    cbar.set_ticks([0, 5, 10, 15, 20,]) # 需要按情况根据最大值进行适当调整。

    new_ticks = np.linspace(-1, 2.5, 8)
    print(new_ticks)
    plt.xticks(new_ticks)


    plt.xlabel("COM Distance/nm",font = "Arial",fontsize=36,weight="bold")
    plt.ylabel("Angle/°",font = "Arial",fontsize=36,weight="bold")
    plt.xticks(font="Arial",size=30,weight="bold")
    plt.yticks(font="Arial",size=30,weight="bold")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    plt.show()
    fig.savefig("Figure_2.pdf")

def main():

    in_file = sys.argv[1]
    process(in_file)

if __name__ == '__main__':
    main()