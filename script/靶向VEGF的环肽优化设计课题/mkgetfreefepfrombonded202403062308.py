import os
import time
import shutil
from collections import defaultdict

class config:

    def __init__(self):

        self.bonded = os.path.join("./bonded_in/system.pdb")
        self.seg = dict({"MUT":"MUT", "SOL":"SOL", "WTA":"WTA", "WTB":"WTB", "WTC":"WTC", "WTD":"WTD", "ION":"ION2", "WT1":"WTE", "WT2":"WTF", "WT3":"WTG", "WT4":"WTH", "WT5":"WTI", "WT6":"WTJ", "WT7":"WTK", "WT8":"WTL", "ION1":"ION1"})
        self.ff_path = os.path.join("/public/home/yqyang/file/vegf-toppar/")
        self.vmd_path = os.path.join("/public/home/yqyang/software/vmd-1.9.4a55-installed/bin/vmd")

def md_pbc_box(pdb, vmd_path):
    mk_pbcbox = open("mk_pbcbox.tcl", "w")
    mk_pbcbox.write(
"""
#!/bin/bash
# vmd -dispdev text -e mk_pbcbox.tcl

package require psfgen
psfcontext reset
mol load pdb {0}
set everyone [atomselect top all]
set minmax [measure minmax $everyone]
foreach {{min max}} $minmax {{ break }}
foreach {{xmin ymin zmin}} $min {{ break }}
foreach {{xmax ymax zmax}} $max {{ break }}

set file [open "PBCBOX.dat" w]
puts $file "{{{{$xmin $ymin $zmin}} {{$xmax $ymax $zmax}}}}"

exit
""".format(pdb)
    )
    mk_pbcbox.close()
    cmd = vmd_path + " -dispdev text -e mk_pbcbox.tcl"
    os.system(cmd)
    time.sleep(1)  

    with open("PBCBOX.dat") as f:
        box = f.readline().strip()

    return box

def split_seg(bonded, seg):

    with open(bonded) as f:
        f1 = f.readlines()
    seg_lines = defaultdict(list)
    for i in f1:
        if i.startswith("ATOM"):
            seg_t = i.strip().split()[-1]
            if seg_t in seg.keys():
                seg_lines[seg[seg_t]].append(i.replace(seg_t, seg[seg_t]))
    
    for i in seg_lines.keys():
        rt = open(i + ".pdb", "w")
        rt.write("".join(seg_lines[i]))
        rt.close()

    return seg_lines

def build_free(seg_lines, ff_path, box_size, vmd_path):

    mk_build = open("mk_build.tcl", "w")
    mk_build.write(
'''
package require psfgen
psfcontext reset
topology {0}/top_all36_prot.rtf
topology {0}/toppar_all36_prot_c36m_d_aminoacids.str
topology {0}/toppar_water_ions.str

segment MUT {{
    first none
    last CT2
    pdb MUT.pdb
    }}
'''.format(ff_path)
    )

    new_seg = []
    for i in seg_lines.keys():
        if i != "MUT":
            new_seg.append(i)

    for i in new_seg:
        mk_build.write(
'''
segment {0} {{
    first none
    last none
    pdb {0}.pdb
    }}
'''.format(i)
        )

    mk_build.write(
'''
patch DISU MUT:1 MUT:11
patch CONH MUT:10 MUT:14
regenerate angles dihedrals
'''
    )

    for i in seg_lines.keys():
        mk_build.write(
'''
coordpdb {0}.pdb {0}
'''.format(i)
        )

    mk_build.write(
'''
guesscoord
writepdb merged.pdb
writepsf merged.psf

psfcontext reset
mol load psf merged.psf pdb merged.pdb
package require solvate
solvate merged.psf merged.pdb -minmax {0} -o solvated
mol delete all
package require autoionize
autoionize -psf solvated.psf -pdb solvated.pdb -sc 0.15 -o system
exit
'''.format(box_size)
    )

    mk_build.close()
    cmd = vmd_path + " -dispdev text -e mk_build.tcl"
    os.system(cmd)
    time.sleep(1) 

def clear(seg_lines):

    os.remove("PBCBOX.dat")
    os.remove("mk_pbcbox.tcl")
    for i in seg_lines.keys():
        os.remove(i + ".pdb")
    os.remove("mk_build.tcl")
    os.remove("merged.pdb")
    os.remove("merged.psf")
    os.remove("solvated.pdb")
    os.remove("solvated.psf")
    if os.path.exists(os.path.join(".", "free_in")):
        shutil.rmtree(os.path.join(".", "free_in"))
        os.makedirs(os.path.join(".", "free_in"))
    else:
        os.makedirs(os.path.join(".", "free_in"))
    shutil.move("system.psf", os.path.join(".", "free_in"))
    shutil.move("system.pdb", os.path.join(".", "free_in"))

def run():
    settings = config()
    box_size = md_pbc_box(settings.bonded, settings.vmd_path)
    seg_lines = split_seg(settings.bonded, settings.seg)
    build_free(seg_lines, settings.ff_path, box_size, settings.vmd_path)
    clear(seg_lines)

def main():
    run()

if __name__=="__main__":
    main() 