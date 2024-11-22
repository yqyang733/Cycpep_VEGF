import os
import shutil

class config:

    def __init__(self):

        self.wins = "wins.dat"

def generate_plumed(wins):

    windows = []
    with open(wins) as f:
        f1 = f.readlines()
    for i in f1:
        windows.append(i.replace("\n", ""))
    all_num = len(windows)
    
    for i in range(all_num):
        job = open(os.path.join(".", windows[i], "plumed.dat"), "w")
        job.write(
'''target: GROUP NDX_FILE=../../index.ndx NDX_GROUP=target
PGLphe: GROUP NDX_FILE=../../index.ndx NDX_GROUP=PGL-phe
angle: GROUP NDX_FILE=../../index.ndx NDX_GROUP=angle

ref_pos: COM ATOMS=target
phe_pos: COM ATOMS=PGLphe
d: DISTANCE ATOMS=ref_pos,phe_pos

ang: ANGLE ATOMS=angle

metad: METAD ARG=d,ang ...
  PACE=250 HEIGHT=1.2 BIASFACTOR=15
  SIGMA=0.2,0.2
  FILE=HILLS GRID_MIN=0,-pi GRID_MAX=5,pi
  WALKERS_N={0}
  WALKERS_ID={1}
  WALKERS_DIR=../../all_walkers/
  WALKERS_RSTRIDE=100
...

PRINT ARG=d,ang FILE=COLVAR STRIDE=10
'''.format(str(all_num), str(i))
        )
        job.close()

def main():

    settings = config()  
    generate_plumed(settings.wins)   

if __name__ == '__main__':
    main()
