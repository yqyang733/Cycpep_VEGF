import os
import sys
from pymol import cmd

def run(wins):

    all_wins = []
    with open(wins) as f:
        f1 = f.readlines()

    for i in f1:
        all_wins.append(i.replace("\n", ""))

    cmd.delete("all")

    for i in all_wins:
        cmd.load(os.path.join(".", i, i+".gro"))
    cmd.remove('solvent')
    cmd.remove('inorganic')
    cmd.remove('resn CLA')
    cmd.remove('resn POPC')
    cmd.remove('resn POPE')
    cmd.remove('resn POPA')
    cmd.remove('resn POPS')
    cmd.remove('resn POCL1')

    for i in range(1, len(all_wins)):
        cmd.align(all_wins[i], all_wins[0])
        cmd.remove("{} and polymer.protein".format(all_wins[i]))

    cmd.save("show.pse",state=0)

def main():

    wins = sys.argv[1]
    run(wins)

if __name__ == '__main__':
    main()