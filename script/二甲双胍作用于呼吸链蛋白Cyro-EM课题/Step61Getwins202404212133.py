import sys
import numpy as np

def compute(dist, step):
    with open(dist) as f:
        f1 = f.readlines()
    data_all = []
    distance = []
    for i in f1:
        if not i.startswith(("#","@")):
            data_all.append(i)
            distance.append(i.strip().split()[1])
    distance = np.array(distance, dtype=np.float64)
    max_dis = max(distance)
    min_dis = min(distance)
    temp = min_dis
    index_need = []
    while temp <= max_dis:
        index_min = np.argmin(abs(distance - temp))
        index_need.append(index_min)
        temp = temp + step

    pick = []
    for i in index_need:
        pick.append(data_all[i])
    result = open("gro_pick.dat", "w")
    result.write("".join(pick).strip())
    time_pick = open("time_pick.dat", "w")
    for i in pick:
        time_pick.write(str(int(float(i.strip().split()[0]))) + "\n")

def main():

    dist = sys.argv[1]   # dist.xvg
    step = float(sys.argv[2])   # 0.025
    compute(dist, step)
    
if __name__=="__main__":
    main()  