import os
import sys
import numpy as np

def PCC_calculation(f_input):

    """
    输入数据格式：
    第一行：name,label,predict
    内容：I3C,0.5,0.52
    """

    with open(f_input) as f:
        f.readline()
        f1 = f.readlines()

    label = []
    predict = []
    
    for i in f1:
        line = i.replace("\n", "").split(",")
        label.append(float(line[1]))
        predict.append(float(line[2]))

    pcc = np.corrcoef(np.array(label), np.array(predict))[(0,1)]

    pcc_rt = open("PCC.dat", "w")
    pcc_rt.write("PCC = " + str(pcc))
    pcc_rt.close()

def plot_traindata_predictdata_scatter(f_input):

    """
    该函数将训练集中所有构象的label以及predict画成散点图。
    在训练集的散点图上将测试集的突变的label和预测画成颜色不一样的散点图。
    数据格式如下：
    predict_train,labels_train,predict_test,labels_test
    0.570176547,0.227529067,0.029530072,0.52
    """

    from matplotlib import cm,colors
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure, show, rc
    import numpy as np
    import pandas as pd

    #%matplotlib inline                   
    plt.rcParams["font.sans-serif"]='SimHei'   #解决中文乱码问题
    plt.rcParams['axes.unicode_minus']=False   #解决负号无法显示的问题
    plt.rc('axes',axisbelow=True)  
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    # ax = plt.gca()

    df=pd.read_csv(f_input)
    #df=df.sort_values(by='1996', ascending=False)

    markers=['*','o','^']                                                                              
    colors=["#66889E","#757CBB","#757CBB"]     

    y1=np.array(df["predict_train"])
    x1=np.array(df["labels_train"])
    y2=np.array(df["predict_test"])
    x2=np.array(df["labels_test"])

    fig=plt.figure(figsize=(5,5))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)           #设置绘图区域大小位置

    m=np.arange(-2.5,2.5,0.1)
    plt.plot(m, m, color="black", lw=2, linestyle='-')  
                                               
    plt.scatter(x1, y1,                                                                  
                s=10, linewidths=0, edgecolors="k",alpha=0.4,                                          
                marker=markers[0], c=colors[0], label="Train")                     
                                                              
    plt.scatter(x2, y2,                                                                  
                s=50, linewidths=0, edgecolors="k",alpha=1,                                            
                marker=markers[1], c="red", label="Test")         

    plt.legend(loc=(0.13,0.9),ncol=2,frameon=False,prop={'family':'Arial','weight':'bold','size':'15'})    #显示图例，loc图例显示位置(可以用坐标方法显示），ncol图例显示几列，默认为1列,frameon设置图形边框

    plt.yticks(font="Arial",size=15,weight="bold")     
    plt.xticks(font="Arial",size=15,weight="bold")                                        #设置y轴刻度，位置,大小
    #plt.grid(axis="y",c=(217/256,217/256,217/256))        #设置网格线
                     #将y轴网格线置于底层
    #plt.xlabel("Quarter",labelpad=10,size=18,)                          #设置x轴标签,labelpad设置标签距离x轴的位置
    #plt.ylabel("Amount",labelpad=10,size=18,)                                   #设置y轴标签,labelpad设置标签距离y轴的位置
    plt.xlabel('Label (ΔΔG)', fontproperties="Arial",fontsize=18,weight="bold")
    plt.ylabel('Predict (ΔΔG)', fontproperties="Arial",fontsize=18,weight="bold")
    # plt.ylim(0, 1.8)

    ax = plt.gca()                         #获取整个表格边框
    #ax.spines['top'].set_color('none')  # 设置上‘脊梁’为无色
    #ax.spines['right'].set_color('none')  # 设置右‘脊梁’为无色
    #ax.spines['left'].set_color('none')  # 设置左‘脊梁’为无色
    plt.show()

    fig.savefig('Figure.pdf')

def remove_traj_water_ions(i):

    ionstcl = open(os.path.join(i, "ionstcl"), "w")
    ionstcl.write("""mol new complex.psf waitfor all
mol addfile com-prodstep.dcd waitfor all

set sel_save [atomselect top "not (resname TIP3 or resname SOD or resname CLA)"]

$sel_save writepsf complex.psf

animate write dcd com-prodstep.dcd sel $sel_save beg 0 end 999 skip 1 0

quit
""")
    ionstcl.close()

    pbctcl = open(os.path.join(i, "pbctcl"), "w")
    pbctcl.write("""package require pbctools
mol new complex.psf waitfor all
mol addfile com-prodstep.dcd waitfor all

pbc wrap -all -compound segid -center com -centersel "protein"

animate write dcd com-prodstep.dcd

quit
""")
    pbctcl.close()

    command_ions = "/public/home/yqyang/software/vmd1.9.3-install/bin/vmd -dispdev text -e ionstcl"
    command_pbc = "/public/home/yqyang/software/vmd1.9.3-install/bin/vmd -dispdev text -e pbctcl"
    os.chdir(i)
    os.system(command_ions)
    os.system(command_pbc)
    os.remove("ionstcl")
    os.remove("pbctcl")
    os.chdir("..")

def main():

    from concurrent.futures import ProcessPoolExecutor

    file = str(sys.argv[1])
    # plot_traindata_predictdata_scatter(file)
    # PCC_calculation(file)
    with open(file) as f:
        f1 = f.readlines()

    all_lst = [i.replace("\n", "") for i in f1]

    with ProcessPoolExecutor(max_workers=int(25)) as executor:
        executor.map(remove_traj_water_ions, all_lst)

    
if __name__=="__main__":
    main() 