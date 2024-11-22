import os
import sys

def process(in_file):
    with open(in_file) as f:
        f1 = f.readlines()

    hillis = []
    for i in f1:
        hillis.append(i.strip())

    for i in hillis:
        rt = open(i, "w")
        with open(os.path.join("..", i)) as f:
            f1 = f.readlines()
        for a in f1:
            if a.startswith("#"):
                rt.write(a)
            else:
                line = a.split()
                if float(line[0]) <= 30000:
                    rt.write(a)
        rt.close()

def main():

    in_file = sys.argv[1]
    process(in_file)

if __name__ == '__main__':
    main()
