from pymol import cmd
from collections import Counter
import math

def contact(ref, traj):
    name_ = "new"
    cmd.load(ref, name_)
    cmd.load_traj(traj)
    num_states = cmd.count_states(name_)
    all_contact_resi = []
    for i in range(num_states):
        myspace = {"lst":[]}
        cmd.select("(br. all within 5 of resn LIG) and (not resn LIG)",state=i+1)
        cmd.iterate("sele","lst.append(resi)",space = myspace)
        all_contact_resi.extend(set(myspace["lst"]))
    tongji = Counter(all_contact_resi)
    tongji = dict(sorted(tongji.items(), key=lambda item: item[1], reverse=True))
    result = open("frequency-5.txt","w")
    for i in tongji:
        result.write(str(i) + "," + str(round(tongji[i]/num_states,3)) + "\n")
    result.close()

contact("../pbc/new.pdb", "../pbc/md_pbcfit_all_new.xtc")
