import os
import time
from collections import defaultdict

class config:

    def __init__(self):

        self.pdb_path = os.path.join("/public/home/yqyang/VEGF/FEP/bonded_in/system.pdb")
        self.psf_path = os.path.join("/public/home/yqyang/VEGF/FEP/bonded_in/system.psf")
        self.segname = "PEP"
        # self.resid = "2"
        self.mutation =  ["2ALA", "3ALA"]
        self.vmd_path = os.path.join("/public/home/yqyang/software/vmd-1.9.4a55-installed/bin/vmd")
        self.ff_path = os.path.join("/public/home/yqyang/file/vegf-toppar")

class Lig_pdb:

    def __init__(self, lig_pdb_file):

        self.segname = set() # all segment names
        self.segresid_atoms = defaultdict(list)  # key: (seg, resid)   value: [atoms,]

        with open(lig_pdb_file) as f:
            f1 = f.readlines()
        for i in f1:
            if i.startswith("ATOM"):
                self.segname.add(i[66:76].strip())
                self.segresid_atoms[(i[66:76].strip(), i[6:11].strip())].append(i[12:16].strip())

def do_mutate(pdb, segname, ff_path, mutation, vmd_path):

    resid_1 = mutation[0][:-3]
    mut_1 = mutation[0][-3:]
    resid_2 = mutation[1][:-3]
    mut_2 = mutation[1][-3:]

    mk_mut = open("mk_mutate.tcl", "w")
    mk_mut.write(
'''
package require psfgen

mol new {0}
set sel [atomselect top "segname {1}"]
$sel writepdb mutant.pdb

resetpsf
topology {6}/top_all36_prot.rtf
topology {6}/toppar_water_ions.str

segment MUT {{
  first none
  last CT2
  pdb mutant.pdb
  mutate {2} {3}
  mutate {4} {5}
}}
patch DISU MUT:1 MUT:11
patch CONH MUT:10 MUT:14
coordpdb mutant.pdb MUT
regenerate angles dihedrals
guesscoord

writepsf mutant.psf
writepdb mutant.pdb

quit
'''.format(pdb, segname, resid_1, mut_1, resid_2, mut_2, ff_path)
    )
    mk_mut.close()
    cmd = vmd_path + " -dispdev text -e mk_mutate.tcl"
    os.system(cmd)
    time.sleep(1)

def do_merge(pdb, psf, segname, vmd_path):

    pdb_seg = Lig_pdb(pdb)
    pdb_seg = [i for i in pdb_seg.segname if i != segname]
    pdb_seg = " ".join(pdb_seg) 
    mk_merge = open("mk_merge.tcl", "w")
    mk_merge.write(
'''
package require topotools

mol new mutant.psf
mol addfile mutant.pdb
mol new {0}
mol addfile {1}

set sel1 [atomselect 0 all]
set sel2 [atomselect 1 "segname {2}"]
set mol [::TopoTools::selections2mol "$sel1 $sel2"]
animate write psf complex.psf $mol
animate write pdb complex.pdb $mol
    
quit
'''.format(psf, pdb, pdb_seg)
    )
    mk_merge.close()
    cmd = vmd_path + " -dispdev text -e mk_merge.tcl"
    os.system(cmd)
    time.sleep(1)
    
def run():
    import sys
    settings = config()
    settings.mutation = sys.argv[1:]
    do_mutate(settings.pdb_path, settings.segname, settings.ff_path, settings.mutation, settings.vmd_path)
    do_merge(settings.pdb_path, settings.psf_path, settings.segname, settings.vmd_path)
    os.remove("mk_mutate.tcl")
    os.remove("mk_merge.tcl")
    os.remove("mutant.psf")
    os.remove("mutant.pdb")

def main():
    run()

if __name__=="__main__":
    main() 
