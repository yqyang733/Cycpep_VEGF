import sys

def process(file_in):
    with open(file_in) as f:
        f1 = f.readlines()
    x_A = []
    y_A = []
    z_A = []
    x_B = []
    y_B = []
    z_B = []
    for i in f1:
        if i.startswith("ATOM") and i[21:22] == "A":
            x_A.append(float(i[30:38]))
            y_A.append(float(i[38:46]))
            z_A.append(float(i[46:54]))
        if i.startswith("ATOM") and i[21:22] == "B":
            x_B.append(float(i[30:38]))
            y_B.append(float(i[38:46]))
            z_B.append(float(i[46:54]))
    # center_A = ((max(x_A)+min(x_A))/2, (max(y_A)+min(y_A))/2, (max(z_A)+min(z_A))/2)
    center_A = (50.427,52.181,49.394)   # 该位置是蛋白质心的坐标，在pymol中由centerofmass获得。
    dev = 1000
    id_ = 0
    for a in range(len(x_A)):
        dist = (x_A[a]-center_A[0])**2 + (y_A[a]-center_A[1])**2 + (z_A[a]-center_A[2])**2
        if dist <= dev:
            id_ = a+1
            dev = dist
    print("A:", id_)

    center_B = ((max(x_B)+min(x_B))/2, (max(y_B)+min(y_B))/2, (max(z_B)+min(z_B))/2)
    dev_b = 1000
    id_b = 0
    for a in range(len(x_B)):
        dist_b = (x_B[a]-center_B[0])**2 + (y_B[a]-center_B[1])**2 + (z_B[a]-center_B[2])**2
        if dist_b <= dev_b:
            id_b = a+1
            dev_b = dist
    print("B:", id_b + len(x_A))

def main():
    file_in = sys.argv[1]
    process(file_in)

if __name__ == '__main__':
    main()