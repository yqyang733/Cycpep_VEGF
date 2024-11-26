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