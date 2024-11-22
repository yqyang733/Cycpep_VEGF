from matplotlib import cm,colors
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show, rc
import numpy as np
import pandas as pd

time = []
dup_1 = []
dup_2 = []
dup_3 = []

with open("entrance_run1.xvg") as f:
    f1 = f.readlines()
for i in f1:
    if i.startswith("@") or i.startswith("#"):
        pass
    else:
        line = i.strip().split()
        if float(line[0]) <= 200000:
            time.append(float(line[0])/1000)
            dup_1.append(float(line[1]))

with open("entrance_run2.xvg") as f:
    f1 = f.readlines()
for i in f1:
    if i.startswith("@") or i.startswith("#"):
        pass
    else:
        line = i.strip().split()
        if float(line[0]) <= 200000:
            # time.append(float(line[0])/1000)
            dup_2.append(float(line[1]))

with open("entrance_run3.xvg") as f:
    f1 = f.readlines()
for i in f1:
    if i.startswith("@") or i.startswith("#"):
        pass
    else:
        line = i.strip().split()
        if float(line[0]) <= 200000:
            # time.append(float(line[0])/1000)
            dup_3.append(float(line[1]))

# df = pd.read_csv(file)
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.3)
ax = plt.gca()
# df['Time'] = df['Time']/1000
b1, = plt.plot(time,dup_1,linewidth=2, label="Run1",color="#66889E")
b2, = plt.plot(time,dup_2,linewidth=2, label="Run2",color="#C35C6A")
b3, = plt.plot(time,dup_3,linewidth=2, label="Run3",color="#7E527F")
# b4, = plt.plot(df['frames']/2,df['pro']/10,linewidth=2, label="protein")
# b1, = plt.plot(df['frames']/2,df['lig']/10,linewidth=2, label="lig")
# b2, = plt.plot(df['frames']/2,df['pkt']/10,linewidth=2, label="pkt")    

plt.xlabel('Time (ns)', fontproperties="Arial",fontsize=24,weight="bold")
plt.ylabel('Angle (°)', fontproperties="Arial",fontsize=24,weight="bold")
# plt.ylabel('Frequency',fontproperties="Arial",fontsize=28,weight="bold")   # 设置y轴标签
plt.xticks(font="Arial",rotation=0,size=18,weight="bold")      # size must be after the font.
plt.yticks(font="Arial",size=18,weight="bold")
# plt.title('Frequency_vdw', fontproperties='Arial', fontsize=33)   # 设置图片标题
plt.legend(handles=[b1,b2,b3,],loc=(0.76,0.7),ncol=1,frameon=False,prop="Arial")    #显示图例，loc图例显示位置(可以用坐标方法显示），ncol图例显示几列，默认为1列,frameon设置图形边框
# plt.legend(handles=[b1,b2,b4],loc=(0.46,0.84),ncol=2,frameon=False,prop="Arial")
# plt.ylim(0, 0.5)
plt.ylim(0, 60)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=15, weight="bold")
plt.show()
fig.savefig('huitu.pdf')
