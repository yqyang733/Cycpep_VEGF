class force_field:
    
    def __init__(self, force_itp):
        
        self.bondtypes_lst = list()
        self.angletypes_lst = list()
        self.dihedraltypes_lst = list()
        self.bondtypes = dict()
        self.angletypes = dict()
        self.dihedraltypes = dict()

        index = 0
        flag_mol = ""

        with open(force_itp) as f:
            lines = f.readlines()

        new_lines = []
        for i in lines:
            if i.startswith(";") or i.startswith("#"):
                pass
            else:
                new_lines.append(i.strip())
        lines = new_lines

        while index < len(lines):
            line = lines[index]
            
            if line.startswith("[") and line.strip().endswith("]"):
                flag_mol = line
            if flag_mol == "[ bondtypes ]":
                self.bondtypes_lst.append(line)
            elif flag_mol == "[ angletypes ]":
                self.angletypes_lst.append(line)
            elif flag_mol == "[ dihedraltypes ]":
                self.dihedraltypes_lst.append(line)

            index += 1

        # print("self.bondtypes_lst", self.bondtypes_lst)

        for i in self.bondtypes_lst:
            if i.startswith("[") or i=="":
                pass
            else:
                # print(i)
                line = i.split()
                bond_type = (line[0], line[1], line[2])
                self.bondtypes[bond_type] = i

        for i in self.angletypes_lst:
            if i.startswith("[") or i=="":
                pass
            else:
                line = i.split()
                angle_type = (line[0], line[1], line[2], line[3])
                self.angletypes[angle_type] = i

        for i in self.dihedraltypes_lst:
            if i.startswith("[") or i=="":
                pass
            else:
                # print(i)
                line = i.split()
                dihedral_type = (line[0], line[1], line[2], line[3], line[4])
                self.dihedraltypes[dihedral_type] = i

def find_bond(ddict, bond_type):
    
    bond_type_1 = bond_type
    bond_type_2 = (bond_type[1], bond_type[0], bond_type[2])
    try:        
        print(ddict[bond_type_1])
    except:
        print("not found: ", bond_type_1)

    try:        
        print(ddict[bond_type_2])
    except:
        print("not found: ", bond_type_2)

def find_angle(ddict, angle_type):
    
    angle_type_1 = angle_type
    angle_type_2 = (angle_type[2], angle_type[1], angle_type[0], angle_type[3])
    try:        
        print(ddict[angle_type_1])
    except:
        print("not found: ", angle_type_1)

    try:        
        print(ddict[angle_type_2])
    except:
        print("not found: ", angle_type_2)

def find_dihe(ddict, dihe_type):
    
    dihe_type_1 = dihe_type
    dihe_type_2 = (dihe_type[3], dihe_type[2], dihe_type[1], dihe_type[0], dihe_type[4])
    try:        
        print(ddict[dihe_type_1])
    except:
        print("not found: ", dihe_type_1)

    try:        
        print(ddict[dihe_type_2])
    except:
        print("not found: ", dihe_type_2)

def find_impro(ddict, impro_type):
    
    impro_type_1 = impro_type
    impro_type_2 = (impro_type[0], impro_type[1], impro_type[3], impro_type[2], impro_type[4])
    impro_type_3 = (impro_type[0], impro_type[2], impro_type[1], impro_type[3], impro_type[4])
    impro_type_4 = (impro_type[0], impro_type[2], impro_type[3], impro_type[1], impro_type[4])
    impro_type_5 = (impro_type[0], impro_type[3], impro_type[1], impro_type[2], impro_type[4])
    impro_type_6 = (impro_type[0], impro_type[3], impro_type[2], impro_type[1], impro_type[4])
    try:        
        print(ddict[impro_type_1])
    except:
        print("not found: ", impro_type_1)

    try:        
        print(ddict[impro_type_2])
    except:
        print("not found: ", impro_type_2)

    try:        
        print(ddict[impro_type_3])
    except:
        print("not found: ", impro_type_3)

    try:        
        print(ddict[impro_type_4])
    except:
        print("not found: ", impro_type_4)

    try:        
        print(ddict[impro_type_5])
    except:
        print("not found: ", impro_type_5)

    try:        
        print(ddict[impro_type_6])
    except:
        print("not found: ", impro_type_6)

bondtypes_dict = force_field("ffbonded.itp").bondtypes
angletypes_dict = force_field("ffbonded.itp").angletypes
dihedraltypes_dict = force_field("ffbonded.itp").dihedraltypes
flag = "dihe"  # bond, angle, dihe, impro
tpppye = ("C","NH1","CT2","CT2","9") 
if flag == "bond":
    find_bond(bondtypes_dict, tpppye)
if flag == "angle":
    find_angle(angletypes_dict, tpppye)
if flag == "dihe":
    find_dihe(dihedraltypes_dict, tpppye)
if flag == "impro":
    find_impro(dihedraltypes_dict, tpppye)