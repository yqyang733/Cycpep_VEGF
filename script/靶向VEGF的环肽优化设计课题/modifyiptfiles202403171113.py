import os
from collections import defaultdict

class Lig_itp:
    
    def __init__(self, lig_itp_file):
        
        self.moleculetype = list()
        self.atoms = list()
        self.bonds = list()
        self.pairs = list()
        self.angles = list()
        self.dihedrals = list()
        self.imp = list()
        self.cmap = list()

        index = 0
        dihe_index = 0
        flag_mol = ""
        with open(lig_itp_file) as f:
            lines = f.readlines()

        self.title = lines[:24]
        self.end = lines[-4:]

        while index < len(lines):
            line = lines[index]
            
            if line.startswith("[") and line.strip().endswith("]"):
                flag_mol = line.strip()
                if flag_mol == "[ dihedrals ]":
                    dihe_index += 1
            if flag_mol == "[ moleculetype ]":
                self.moleculetype.append(line)
            elif flag_mol == "[ atoms ]":
                self.atoms.append(line)
            elif flag_mol == "[ bonds ]":
                self.bonds.append(line)
            elif flag_mol == "[ pairs ]":
                self.pairs.append(line)                
            elif flag_mol == "[ angles ]":
                self.angles.append(line)
            elif flag_mol == "[ dihedrals ]":
                if dihe_index == 1:
                    self.dihedrals.append(line)
                else:
                    self.imp.append(line)
                
            elif flag_mol == "[ cmap ]":
                self.cmap.append(line)

            index += 1
        
    def get_residues(self):

        self.resi_lines = defaultdict(list)
        for i in self.atoms:
            if i.startswith("[") or i.startswith(";   nr") or i.startswith("\n"):
                pass
            else:
                if i.startswith("; residue"):
                    line = i.split()
                    resi = line[2]
                    self.resi_lines[resi].append(i)
                else:
                    self.resi_lines[resi].append(i)
        return self.resi_lines

need_del = ["HZ1", "HZ2"]
need_bond = [((10,14),("CD","NZ"),(1,0.1345,309616)),]
need_angle = [((10,10,14),("OE1","CD","NZ"),(5,124,669.44,0,0)),((10,10,14),("CG","CD","NZ"),(5,112.5,167.36,0,0)),((10,14,14),("CD","NZ","CE"),(5,120,418.4,0,0)),((10,14,14),("CD","NZ","HZ3"),(5,120,418.4,0,0))]
need_dihe = [((10,10,14,14),("OE1","CD","NZ","CE"),(9,180,10.46,2)),((10,10,14,14),("OE1","CD","NZ","HZ3"),(9,180,10.46,2)),((10,10,10,14),("CB","CG","CD","NZ"),(9,180,0.2092,6)),((10,10,10,14),("HG1","CG","CD","NZ"),(9,180,0,3)),((10,10,10,14),("HG2","CG","CD","NZ"),(9,180,0,3)),((10,10,14,14),("CG","CD","NZ","CE"),(9,180,10.8784,2)),((10,10,14,14),("CG","CD","NZ","HZ3"),(9,180,5.8576,2)),((10,14,14,14),("CD","NZ","CE","HE1"),(9,0,0,3)),((10,14,14,14),("CD","NZ","CE","HE2"),(9,0,0,3)),((10,14,14,14),("CD","NZ","CE","CD"),(9,0,7.5312,1))]
need_impro = [((10,10,10,14),("CD","CG","OE1","NZ"),(2,0,1004.16)),((14,10,14,14),("NZ","CD","HZ3","CE"),(2,0,167.36))]

itp = Lig_itp("topol_Protein_chain_M.itp")
rt = open("topol_Protein_chain_M_1.itp", "w")
rt.write("".join(itp.title)+"\n")
rt.write("".join(itp.moleculetype))
resi_lines = itp.get_residues()

new_resi10 = list()
new_resi10_dict = dict()
for i in resi_lines["10"]:
    if i.startswith(";"):
        pass
    else:
        line = i.split()
        new_resi10_dict[line[4]] = float(line[6])
new_resi10_dict["CD"] = 0.51
new_resi10_dict["OE1"] = -0.51
total_ele = sum(new_resi10_dict.values())
for i in resi_lines["10"]:
    if i.startswith(";"):
        new_resi10.append(i)  
    else:
        line = i.split()
        new_resi10.append("{:6}   {:>8}   {:>4} {:>6}   {:>4} {:>6} {:>10}   {:>8}\n".format(line[0],line[1],line[2],line[3],line[4],line[5],str(new_resi10_dict[line[4]]),line[7]))
# new_resi10[-1] = new_resi10[-1].split(";")[0] + "; qtot " + str(round(total_ele, 2)) +"\n"
resi_lines["10"] = new_resi10
# print(new_resi10)

new_resi14 = list()
del_id = list()
for i in resi_lines["14"]:
    if i.startswith(";"):
        new_resi14.append(i)
    else:
        line = i.split()
        if line[4] in need_del:
            del_id.append(line[0])
        elif line[4] == "NZ":
            new_resi14.append("{:6}   {:>8}   {:>4} {:>6}   {:>4} {:>6} {:>10}   {:>8}\n".format(line[0],line[1],line[2],line[3],line[4],line[5],str(-0.47),line[7]))
        else:
            new_resi14.append(i)
resi_lines["14"] = new_resi14

rt.write("[ atoms ]\n")
rt.write(";   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n")
index_id = 1
dict_resiname_id = dict()
dict_resiname_id_old = dict()
old2new_id = dict()
for i in resi_lines.keys():
    lines = resi_lines[i]
    for a in lines:
        if a.startswith(";") or a.startswith("\n"):
            rt.write(a)
        else:
            a_lines = a.split()
            dict_resiname_id[(i,a_lines[4])] = index_id
            dict_resiname_id_old[(i,a_lines[4])] = a_lines[0]
            old2new_id[a_lines[0]] =str(index_id)
            rt.write("{:6} ".format(index_id)+a[7:])
            index_id += 1

bond_ids = [dict_resiname_id_old[(str(need_bond[0][0][0]),need_bond[0][1][0])], dict_resiname_id_old[(str(need_bond[0][0][1]),need_bond[0][1][1])]]

for i in itp.bonds:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("["):
        rt.write(i)
    elif i.startswith("\n"):
        pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if bond_ids[0] in line and (bond_ids[1] in line):
            flag = False
        if flag:
            rt.write("{:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], line[2]))
for i in need_bond:
    rt.write("{:<6} {:<6} {:6}   {:.6f}   {:.2f}\n".format(dict_resiname_id[(str(i[0][0]), i[1][0])], dict_resiname_id[(str(i[0][1]), i[1][1])], str(i[2][0]), i[2][1], i[2][2]))
rt.write("\n")

for i in itp.pairs:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("[") or i.startswith("\n"):
        rt.write(i)
    # elif i.startswith("\n"):
    #     pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if bond_ids[0] in line and (bond_ids[1] in line):
            flag = False
        if flag:
            rt.write("{:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], line[2]))

for i in itp.angles:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("["):
        rt.write(i)
    elif i.startswith("\n"):
        pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if bond_ids[0] in line and (bond_ids[1] in line):
            flag = False
        if flag:
            rt.write("{:6} {:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], old2new_id[line[2]], line[3]))
for i in need_angle:
    rt.write("{:<6} {:<6} {:<6} {:6}   {:.6f}   {:.6f}   {:.8f}   {:.2f}\n".format(dict_resiname_id[(str(i[0][0]), i[1][0])], dict_resiname_id[(str(i[0][1]), i[1][1])], dict_resiname_id[(str(i[0][2]), i[1][2])], str(i[2][0]), i[2][1], i[2][2], i[2][3], i[2][4]))
rt.write("\n")

for i in itp.dihedrals:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("["):
        rt.write(i)
    elif i.startswith("\n"):
        pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if bond_ids[0] in line and (bond_ids[1] in line):
            flag = False
        if flag:
            rt.write("{:6} {:6} {:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], old2new_id[line[2]], old2new_id[line[3]], line[4]))
for i in need_dihe:
    rt.write("{:<6} {:<6} {:<6} {:<6} {:6}   {:.6f}   {:.6f}   {:6}\n".format(dict_resiname_id[(str(i[0][0]), i[1][0])], dict_resiname_id[(str(i[0][1]), i[1][1])], dict_resiname_id[(str(i[0][2]), i[1][2])], dict_resiname_id[(str(i[0][3]), i[1][3])], str(i[2][0]), i[2][1], i[2][2], str(i[2][3])))
rt.write("\n")

for i in itp.imp:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("["):
        rt.write(i)
    elif i.startswith("\n"):
        pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if bond_ids[0] in line and (bond_ids[1] in line):
            flag = False
        if flag:
            rt.write("{:6} {:6} {:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], old2new_id[line[2]], old2new_id[line[3]], line[4]))
for i in need_impro:
    rt.write("{:<6} {:<6} {:<6} {:<6} {:6}   {:.6f}   {:.6f}\n".format(dict_resiname_id[(str(i[0][0]), i[1][0])], dict_resiname_id[(str(i[0][1]), i[1][1])], dict_resiname_id[(str(i[0][2]), i[1][2])], dict_resiname_id[(str(i[0][3]), i[1][3])], str(i[2][0]), i[2][1], i[2][2]))
rt.write("\n")

for i in itp.cmap:
    flag = True
    print(i)
    if i.startswith(";") or i.startswith("[") or i.startswith("\n") or i.startswith("#"):
        rt.write(i)
    # elif i.startswith("\n"):
    #     pass
    else:
        line = i.strip().split()
        for a in line:
            print(a)
            if a in del_id:
                flag = False
        if flag:
            rt.write("{:6} {:6} {:6} {:6} {:6} {:6}\n".format(old2new_id[line[0]], old2new_id[line[1]], old2new_id[line[2]], old2new_id[line[3]], old2new_id[line[4]], line[5]))

print(del_id)

rt.close()

os.rename("topol_Protein_chain_M.itp", "topol_Protein_chain_M_2.itp")
os.rename("topol_Protein_chain_M_1.itp", "topol_Protein_chain_M.itp")
os.rename("topol_Protein_chain_M_2.itp", "topol_Protein_chain_M_1.itp")

with open("posre_Protein_chain_M.itp") as f:
    f1 = f.readlines()
rt = open("posre_Protein_chain_M_1.itp", "w") 
for i in f1:
    if i.startswith(";") or i.startswith("[") or i.startswith("\n"):
        rt.write(i)
    else:
        print(i)
        line = i.strip().split()
        rt.write("{:>6} {:>6} {:>6} {:>6} {:>6}\n".format(old2new_id[line[0]],line[1],line[2],line[3],line[4],))
rt.close()

os.rename("posre_Protein_chain_M.itp", "posre_Protein_chain_M_2.itp")
os.rename("posre_Protein_chain_M_1.itp", "posre_Protein_chain_M.itp")
os.rename("posre_Protein_chain_M_2.itp", "posre_Protein_chain_M_1.itp")

del_atoms_1 = ("14", "HZ1")
del_atoms_2 = ("14", "HZ2")

with open("build.gro") as f:
    f1 = f.readlines()

rt = open("build_1.gro", "w")

rt.write(f1[0])
rt.write(" "+str(int(f1[1].strip())-2)+"\n")
for i in f1[2:-2]:
    id_atom = i[:5].strip()
    name_atom = i[9:15].strip()
    if (id_atom, name_atom) == del_atoms_1 or ((id_atom, name_atom) == del_atoms_2):
        pass
    else:
        rt.write(i)
rt.write(f1[-2])
rt.write(f1[-1])

os.rename("build.gro", "build_2.gro")
os.rename("build_1.gro", "build.gro")
os.rename("build_2.gro", "build_1.gro")
