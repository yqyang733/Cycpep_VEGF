import os
import sys

def process(in_file):
    with open(in_file) as f:
        f1 = f.readlines()

    rt = open("contsurf_input_2.csv", "w")
    for i in f1:
        line = i.split(",", 1)
        rt.write(str(float(line[0])-1.0)+","+line[1])
    rt.close()

def main():

    in_file = sys.argv[1]
    process(in_file)

if __name__ == '__main__':
    main()