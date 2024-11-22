import sys

start_point = 1.5
end_point = 3.7

data = []

def process(in_file):

    with open(in_file) as f:
        f1 =f.readlines()

    for i in f1:
        if i.startswith("#") or i.startswith("@"):
            pass
        else:
            line = i.strip().split()
            data.append(line)

    data_1 = []
    cha = float(data[0][0]) - start_point
    for i in data:
        data_1.append([float(i[0])-cha, float(i[1]), float(i[2])])

    data_2 = []
    for i in data_1:
        if i[0] <= end_point:
            data_2.append(i)

    data_3 = []
    end_y = data_2[-1][1]
    for i in data_2:
        data_3.append([i[0], i[1]-end_y, i[2]])

    rt = open(in_file+"_1", "w")
    for i in data_3:
        rt.write(str(i[0])+"   "+str(i[1])+"   "+str(i[2])+"\n")
    rt.close()

def main():

    in_file = sys.argv[1]
    process(in_file)
    # 使用方法：python do.py zerror.xvg

if __name__ == '__main__':
    main()  