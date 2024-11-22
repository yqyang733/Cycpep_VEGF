import sys

def process(file_in):
    with open(file_in) as f:
        f1 = f.readlines()
    x = []
    y = []
    z = []
    for i in f1:
        if i.startswith("ATOM"):
            x.append(float(i[30:38]))
            y.append(float(i[38:46]))
            z.append(float(i[46:54]))
    center = ((max(x)+min(x))/2, (max(y)+min(y))/2, (max(z)+min(z))/2)
    x_com = max(x) - min(x)
    y_com = max(y) - min(y)
    z_com = max(z) - min(z)
    print("center of complex:", center)
    print("x:", x_com)
    print("y:", y_com)
    print("z:", z_com)

def main():
    file_in = sys.argv[1]
    process(file_in)

if __name__ == '__main__':
    main()