import os
import math
import pybel
import pickle
import shutil
import numpy as np
from pymol import cmd
from collections import defaultdict
from config import Config

get_des              =       Config().get_des
rank                 =       Config().rank
threshold            =       float(Config().threshold)

def mk_files(fle):

    if not os.path.exists(fle):
        os.makedirs(fle)
    else:
        shutil.rmtree(fle)
        os.makedirs(fle)

def get_residue_reference():
    return {'LEU', 'MET', 'ILE', 'GLU', 'CYS', 'GLY', 'PHE', 'ASN', 'GLN', 'LYS', 'TYR', 'ARG', 'THR', 'PRO', 'VAL', 'ASP', 'ALA', 'TRP', 'HIS', 'SER', 'NLE', 'HSD', 'HSE', 'HSP'}        

def get_atoms_reference():
    return {'N.3':7, 'O.spc':8, 'O.t3p':8, 'C.3':6, 'O.co2':8, 'N.pl3':7, 'S.2':16, 'O.3':8, 'O.2':8, 'C.cat':6, 'P.3':15, 'C.2':6, 'C.ar':6, 'N.1':7, 'N.ar':7, 'C.1':6, 'S.o2':16, 'N.4':7, 'S.3':16, 'S.o':16, 'N.am':7, 'N.2':7}    

class Pdb(object):
    """ Object that allows operations with protein files in PDB format. """
 
    def __init__(self, file_cont = [], pdb_code = ""):
        self.cont = []
        self.atom = []
        self.hetatm = []
        self.fileloc = ""
        if isinstance(file_cont, list):
            self.cont = file_cont[:]
        elif isinstance(file_cont, str):
            try:
                with open(file_cont, 'r') as pdb_file:
                    self.cont = [row.strip() for row in pdb_file.read().split('\n') if row.strip()]
            except FileNotFoundError as err:
                print(err)
 
        if self.cont:
             self.atom = [row for row in self.cont if row.startswith('ATOM')]
             self.hetatm = [row for row in self.cont if row.startswith('HETATM')]
             self.conect = [row for row in self.cont if row.startswith('CONECT')]
 
    def renumber_atoms(self, start=1):
        """ Renumbers atoms in a PDB file. """
        out = list()
        count = start
        for row in self.cont:
            if len(row) > 5:
                if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
                    num = str(count)
                    while len(num) < 5:
                        num = ' ' + num
                    row = '%s%s%s' %(row[:6], num, row[11:])
                    count += 1
            out.append(row)
        return out
 
 
    def renumber_residues(self, start=1, reset=False):
            """ Renumbers residues in a PDB file. """
            out = list()
            count = start - 1
            cur_res = ''
            for row in self.cont:
                if len(row) > 25:
                    if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
                        next_res = row[22:27].strip() # account for letters in res., e.g., '1A'
                        
                        if next_res != cur_res:
                            count += 1
                            cur_res = next_res
                        num = str(count)
                        while len(num) < 3:
                            num = ' ' + num
                        new_row = '%s%s' %(row[:23], num)
                        while len(new_row) < 29:
                            new_row += ' '
                        xcoord = row[30:38].strip()
                        while len(xcoord) < 9:
                            xcoord = ' ' + xcoord
                        row = '%s%s%s' %(new_row, xcoord, row[38:])
                        if row.startswith('TER') and reset:
                            count = start - 1
                out.append(row)
            return out

def split_pdb(pdb, partA, partB, des_path):

    with open(os.path.join(des_path, pdb)) as f:
        f1 = f.readlines()

    MUT = open(os.path.join(des_path, str(pdb).replace(".pdb", "_l.pdb")), "w")
    PRO = open(os.path.join(des_path, str(pdb).replace(".pdb", "_r.pdb")), "w")

    for i in f1:
        if i.startswith("ATOM"):
            if i[21:22].strip() in partA:
                MUT.write(i)
            elif i[21:22].strip() in partB:
                PRO.write(i)

    MUT.close()
    PRO.close()

    if not os.path.getsize(os.path.join(des_path, str(pdb).replace(".pdb", "_r.pdb"))):
        return str(pdb).replace(".pdb", "_r.mol2"), str(pdb).replace(".pdb", "_l.mol2"), "0"
    else:
        pdb1 = Pdb(os.path.join(des_path, str(pdb).replace(".pdb", "_r.pdb")))
        pdb1.cont = pdb1.renumber_residues(start=1, reset=False)
        PRO = open(os.path.join(des_path, str(pdb).replace(".pdb", "_r_renum.pdb")), "w")
        PRO.write("\n".join(pdb1.cont))
        PRO.close()

        pdb1 = Pdb(os.path.join(des_path, str(pdb).replace(".pdb", "_l.pdb")))
        pdb1.cont = pdb1.renumber_residues(start=1, reset=False)
        MUT = open(os.path.join(des_path, str(pdb).replace(".pdb", "_l_renum.pdb")), "w")
        MUT.write("\n".join(pdb1.cont))
        MUT.close()
                
        mol = next(pybel.readfile("pdb", os.path.join(des_path, str(pdb).replace(".pdb", "_r_renum.pdb"))))
        output = pybel.Outputfile("mol2", os.path.join(des_path, str(pdb).replace(".pdb", "_r.mol2")), overwrite=True)
        output.write(mol)
        output.close()
        mol = next(pybel.readfile("pdb", os.path.join(des_path, str(pdb).replace(".pdb", "_l_renum.pdb"))))
        output = pybel.Outputfile("mol2", os.path.join(des_path, str(pdb).replace(".pdb", "_l.mol2")), overwrite=True)
        output.write(mol)
        output.close()
    
    return str(pdb).replace(".pdb", "_r.mol2"), str(pdb).replace(".pdb", "_l.mol2"), "1"

def get_single_snapshot(work_path, mut, refname, trajname, startframe, endframe, step, partA, partB, errlog):

    '''
    Split the molecular dynamics trajectory file into individual PDB structure files, one frame per file.
    '''

    eror_dict = dict()
    eror_lst = list()
    des_path = os.path.join("Descriptors", mut)
    mk_files(des_path)

    shutil.copy(os.path.join(work_path, mut, refname), os.path.join(des_path, refname))
    shutil.copy(os.path.join(work_path, mut, trajname), os.path.join(des_path, trajname))

    cmd.load(os.path.join(des_path, refname), "structure")
    cmd.load_traj(os.path.join(des_path, trajname))
    
    r_lst = []
    l_lst = []
    r_lst_true = []
    l_lst_true = []    
    for i in range(startframe, endframe, step):
        cmd.select("aaaaaaa", "(br. all within {0} of chain {1})".format(threshold, "+".join(partA)), state = i)
        cmd.save(os.path.join(des_path, "{0}_{1}.pdb".format(mut, str(i))), selection="(aaaaaaa)", state = i)
        r_l, l_l, flag = split_pdb("{0}_{1}.pdb".format(mut, str(i)), partA, partB, des_path)

        r_lst.append(r_l)
        l_lst.append(l_l)
        if flag == "0":
            eror_lst.append((r_l, l_l))
            errlog.write("{0}_{1}.pdb".format(mut, str(i)) + "\n")
            errlog.flush()
        else:
            r_lst_true.append(r_l)
            l_lst_true.append(l_l)

    cmd.delete("all")

    sample = np.random.choice(range(len(list(zip(r_lst_true, l_lst_true)))), len(eror_lst))
    for i in range(len(sample)):
        eror_dict[eror_lst[i]] = tuple(list(zip(r_lst_true, l_lst_true))[sample[i]])

    # mean = float(val)

    # r_lst = ["{0}_{1}_r.mol2".format(mut, str(i)) for i in range(startframe, endframe, step)]
    # l_lst = ["{0}_{1}_l.mol2".format(mut, str(i)) for i in range(startframe, endframe, step)]
    # label = [mean for i in range(startframe, endframe, step)]
    # print("r_lst", r_lst)
    # print("l_lst", l_lst)

    return r_lst, l_lst, des_path, eror_dict

def read_complexes(protein_list, ligand_list, des_path, eror_dict):
    
    # data, labels = dict(), dict()
    data = dict()
    feature_inique = set()
    all_frame_names = []
    
    for idx, (p_file, l_file) in enumerate(zip(protein_list, ligand_list)):
        
        '''
        protein atoms type : Store SYMBL of each atom in list.
        protein atoms type index dict : Store index for nine atom types.
        protein atoms coords : The coordinates of each atom.
        protein bonds : Store each atom's bond in an undirected graph((start, end),(start, end)).        
        '''

        if (p_file, l_file) in eror_dict:
            p_file_tmp = eror_dict[(p_file, l_file)][0]
            l_file_tmp = eror_dict[(p_file, l_file)][1]

            ligand_atoms_type, ligand_atoms_type_index_dict, ligand_atoms_coords, ligand_bonds = read_protein_file((open(os.path.join(des_path, l_file_tmp), "r")).readlines())
            protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = read_protein_file((open(os.path.join(des_path, p_file_tmp), "r")).readlines())
        else:
            ligand_atoms_type, ligand_atoms_type_index_dict, ligand_atoms_coords, ligand_bonds = read_protein_file((open(os.path.join(des_path, l_file), "r")).readlines())  
            protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = read_protein_file((open(os.path.join(des_path, p_file), "r")).readlines())

        '''
        Find all cases where the distance between the ligand and the protein atom is less than a certain threshold.
        interaction list -> key : (protein atom type, ligand atom type), 
        value : [[ligand atom index, protein atom index], [ligand atom index, protein atom index], ...]
        '''
        if get_des == "all":
            interaction_list = get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords)
        elif get_des == "rank":
            interaction_list = get_interaction_min(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords)
        ''' find graph '''
        graphs = r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type)
        
        protein_name = p_file.split("/")[-1].split(".")[0]
        ligand_name = l_file.split("/")[-1].split(".")[0]
        
        name = protein_name + "_" + ligand_name
        all_frame_names.append(name)
        print("graphs", graphs)
        print("get_all_features(graphs)", get_all_features(graphs))
        feature_inique.update(get_all_features(graphs))

        with open(os.path.join(des_path, name + ".pkl"), "wb") as f:
            pickle.dump(graphs, f)
        
    for i in all_frame_names:
        with open(os.path.join(des_path, i + ".pkl"), "rb") as f:
            graphs = pickle.load(f)
        data[i] = make_data(graphs, feature_inique)
        # labels[name] = labels_list[idx]
        
    return feature_inique, data

def read_protein_file(lines):
    
    atoms_reference_dict = get_atoms_reference()
    residue_reference_set = get_residue_reference()
    residue_list, atoms_dict, protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = list(), dict(), list(), dict(), list(), dict()
    flag, index, atom_count = 0, 0, 0    

    while index < len(lines):
        
        ''' Extract information about the atom '''
        line = lines[index].strip()    
    
        if flag == 1:
            if "@<TRIPOS>BOND" not in line:
                line_list = line.split()
                
                ''' Extract SYMBL type of atom '''
                atoms_number = int(line_list[0])
                atom_type = line_list[5]
                residue_name = line_list[7].strip()[:3]    
                
                ''' 
                It works if it is not H(hydrogen).
                Save the coordinate information and atom type, and redefine the index.
                '''    
                if atom_type != "H" and residue_name in residue_reference_set and atom_type in atoms_reference_dict:    

                    protein_atoms_coords.append([line_list[2], line_list[3], line_list[4]])   # list
                    protein_atoms_type.append(atoms_reference_dict[atom_type])   # list
                    
                    if atoms_reference_dict[atom_type] in protein_atoms_type_index_dict: 
                        protein_atoms_type_index_dict[atoms_reference_dict[atom_type]].append(atom_count)
                        
                    else:    
                        protein_atoms_type_index_dict[atoms_reference_dict[atom_type]] = [atom_count]                 # protein_atoms_type_index_dict: keys: atom_type_number   values: [new_positions]
                        
                    atoms_dict[atoms_number] = atom_count
                    atom_count += 1                
    
        elif flag == 2:
            ''' Extract informaction about bond. '''
            
            line_list = line.split()    

            start_atom = int(line_list[1]) 
            end_atom = int(line_list[2])
            bond_type = line_list[3]
            
            if start_atom in atoms_dict and end_atom in atoms_dict:
                protein_bonds[(atoms_dict[start_atom], atoms_dict[end_atom])] = bond_type
                protein_bonds[(atoms_dict[end_atom], atoms_dict[start_atom])] = bond_type    
    
        if "@<TRIPOS>ATOM" in line:
            flag = 1
        elif "@<TRIPOS>BOND" in line:
            flag = 2
        
        index += 1
        
    return protein_atoms_type, protein_atoms_type_index_dict, np.array(protein_atoms_coords, dtype = np.float32), protein_bonds        

'''
The protein and ligand atom pairs that exist with a certain distance are stored in
16 different types.
'''        
def get_interaction_min(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords):
        
    interaction_list = make_interaction_dict()    
    dict_ligatm_proatm = dict()    # 键是（配体原子，受体原子），值是原子类型（i, j）
    multidict_ligatm_pairs = defaultdict(list)   # 键是配体原子ID，值是同一个配体原子与受体原子所形成的所有pairs。    
    
    for i in protein_atoms_type_index_dict.keys():
                
        protein_atom_list = protein_atoms_type_index_dict[i]
        protein_atom_list_coords = protein_atoms_coords[protein_atom_list]
        
        for j in ligand_atoms_type_index_dict.keys():
                        
            ligand_atom_list = ligand_atoms_type_index_dict[j]
            ligand_atom_list_coords = ligand_atoms_coords[ligand_atom_list]
            
            interaction_ligand_atoms_index, interaction_protein_atoms_index, interaction_distance_matrix = cal_distance(ligand_atom_list_coords, protein_atom_list_coords) 

            for ligand_, protein_ in zip(interaction_ligand_atoms_index, interaction_protein_atoms_index):
                
                multidict_ligatm_pairs[ligand_atom_list[ligand_]].append((protein_atom_list[protein_], ligand_atom_list[ligand_], interaction_distance_matrix[ligand_][protein_]))
                dict_ligatm_proatm[(protein_atom_list[protein_], ligand_atom_list[ligand_])] = (i, j)
                # interaction_list[(i,j)].append((ligand_atom_list[ligand_], protein_atom_list[protein_]))

    for key in multidict_ligatm_pairs.keys():

        sorted_data = sorted(multidict_ligatm_pairs[key], key=lambda x: x[2])

        for i in range(min(rank, len(sorted_data))):
            
            interaction_list[dict_ligatm_proatm[(sorted_data[i][0], sorted_data[i][1])]].append((sorted_data[i][1], sorted_data[i][0]))
                        
    return interaction_list

def get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords):
        
        interaction_list = make_interaction_dict()        
        
        for i in protein_atoms_type_index_dict.keys():
                
                protein_atom_list = protein_atoms_type_index_dict[i]
                protein_atom_list_coords = protein_atoms_coords[protein_atom_list]
                
                for j in ligand_atoms_type_index_dict.keys():
                        
                        ligand_atom_list = ligand_atoms_type_index_dict[j]
                        ligand_atom_list_coords = ligand_atoms_coords[ligand_atom_list]
                        
                        interaction_ligand_atoms_index, interaction_protein_atoms_index, interaction_distance_matrix = cal_distance(ligand_atom_list_coords, protein_atom_list_coords) 

                        for ligand_, protein_ in zip(interaction_ligand_atoms_index, interaction_protein_atoms_index):
                                interaction_list[(i,j)].append((ligand_atom_list[ligand_], protein_atom_list[protein_]))
                        
        return interaction_list

def distance_x_y_z(coord_a, coord_b):
    return math.sqrt((float(coord_b[0])-float(coord_a[0]))**2 + (float(coord_b[1])-float(coord_a[1]))**2 + (float(coord_b[2])-float(coord_a[2]))**2)

'''
Find protein and ligand atom pairs within a certain distance.
'''        
def cal_distance(ligand_coords, protein_coords):
        
        lig_pro_matrix = []
        for i in ligand_coords:
            tmp = []
            for j in protein_coords:
                tmp.append(distance_x_y_z(i, j))
            lig_pro_matrix.append(tmp)

        distance = np.array(lig_pro_matrix, dtype=np.float32)
        rows, cols = np.where(distance <= threshold)
                
        return rows, cols, distance

'''
Initialize a dictionary to store 36 interactions (protein-ligand pairs).
'''        
def make_interaction_dict():
        
        interaction_list = dict()
        
        protein_atoms_list = [6, 7, 8, 16]
        ligand_atoms_list = [6, 7, 8, 16]
        
        for i in protein_atoms_list:
            for j in ligand_atoms_list:
                interaction_list[(i,j)] = list()
                        
        return interaction_list

def r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type):    

    '''
    undirect, atom = [atom1, atom2, ...] 
    bond dict => start : [end1, end2, ...]
    '''
    graphs = dict()
    ligand_bond_start_end, protein_bond_start_end = make_bond_dict(ligand_bonds, protein_bonds)
    
    for i in interaction_list.keys():
        
        sub_interaction_list = interaction_list[i]
        graph_list = list()
        
        ''' Extract one protein-ligand interaction(pair) '''
        for val in sub_interaction_list:
            
            sub = dict()
            
            ''' find ligand descriptor ''' 
            cur_ligand_atom_index, cur_ligand_atom = val[0], ligand_atoms_type[val[0]]
            
            visited_ligand, step = set(), 0
            visited_ligand.add(cur_ligand_atom_index)
            
            ''' Find the one-step neighborhoods of the current atom. '''
            next_ligand_atom_list, visited_ligand = get_next_atom(cur_ligand_atom_index, ligand_bond_start_end, visited_ligand)
            
            for next_ligand_atom_index in next_ligand_atom_list:
                bond = ligand_bonds[(cur_ligand_atom_index, next_ligand_atom_index)]
                next_ligand_atom = ligand_atoms_type[next_ligand_atom_index]
                
                if ((1, cur_ligand_atom, next_ligand_atom, bond)) in sub:
                    sub[(1, cur_ligand_atom, next_ligand_atom, bond)] += 1
                else:
                    sub[(1, cur_ligand_atom, next_ligand_atom, bond)] = 1
                    
            ''' find protein descriptor ''' 
            cur_protein_atom_index, cur_protein_atom = val[1], protein_atoms_type[val[1]]
            
            visited_protein, step = set(), 0
            visited_protein.add(cur_protein_atom_index)
            
            ''' Find the one-step neighborhoods of the current atom. '''
            next_protein_atom_list, visited_protein = get_next_atom(cur_protein_atom_index, protein_bond_start_end, visited_protein)
            
            for next_protein_atom_index in next_protein_atom_list:
                bond = protein_bonds[(cur_protein_atom_index, next_protein_atom_index)]
                next_protein_atom = protein_atoms_type[next_protein_atom_index]
                
                if ((0, cur_protein_atom, next_protein_atom, bond)) in sub: 
                    sub[(0, cur_protein_atom, next_protein_atom, bond)] += 1
                else:
                    sub[(0, cur_protein_atom, next_protein_atom, bond)] = 1                
            
            graph_list.append(sub)
        
        graphs[i] = graph_list        
    
    return graphs

'''
Find one-step neighborhoods of the current atom.
'''    
def get_next_atom(current_atom, bond_dict, visited_atoms):

    next_atoms_list = bond_dict[current_atom]
    next_atoms_list = list(set(next_atoms_list) -  visited_atoms)    
    
    if len(next_atoms_list) == 0:
        return [], visited_atoms
    else:
        for i in next_atoms_list:
            visited_atoms.add(i)
            
        return next_atoms_list, visited_atoms

'''
Stores bonds between atoms in a dictionary.
'''    
def make_bond_dict(ligand_bonds, protein_bonds):
    
    ligand_start_end, protein_start_end = dict(), dict()
    
    for i in ligand_bonds.keys():
        start = i[0]
        end = i[1]
        
        if start in ligand_start_end:
            ligand_start_end[start].append(end)
            
        else:    
            ligand_start_end[start] = [end]
            
    for i in protein_bonds.keys():    
        start = i[0]
        end = i[1]
    
        if start in protein_start_end:
            protein_start_end[start].append(end)
        else:
            protein_start_end[start] = [end]
            
    return ligand_start_end, protein_start_end

def remove_files(des_path):

    shutil.rmtree(des_path)

def get_all_features(graphs_dict):

    all_features = dict()
        
    for z in graphs_dict.keys():
        for x in graphs_dict[z]:
            tmp = list()
            for y in x.keys():
                tmp.append((y, x[y]))
            tmp = tuple(sorted(tmp))
            if tmp in all_features:
                all_features[tmp] += 1
            else:
                all_features[tmp] = 1

    all_features = list(all_features.keys())

    return all_features

def make_data(graphs_dict, all_features):    
    
    whole_descriptors = dict()
    
    for type in graphs_dict.keys():
        
        ''' 
        one descriptor check  
        e.g. (16, 16):[{(1, 16, 6, '1'): 2, (0, 16, 6, '1'): 1}, ...]    
        '''
        for descriptor in graphs_dict[type]:
            if tuple(sorted(descriptor.items())) in whole_descriptors:
                whole_descriptors[tuple(sorted(descriptor.items()))] += 1
            else:
                whole_descriptors[tuple(sorted(descriptor.items()))] = 1  
    
    ''' Create a row vector for each complex. '''
    row_vetor = list()
    for selected_descriptor in all_features:
        row_vetor.append(whole_descriptors[selected_descriptor]) if selected_descriptor in whole_descriptors else row_vetor.append(0)
            
    return np.array(row_vetor, dtype = np.float32)    
