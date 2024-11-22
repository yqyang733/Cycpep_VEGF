#####################################################################################
# This .py file is only worked for SARS2_3CL_DL_generator_FEP project.              #
# Author: Yanqing Yang                                                              #
# Email: yanqyang@zju.edu.cn; 1821074995@qq.com                                     #
#####################################################################################

#####################################################################################
# Plot RMSD curves with NAMD dcd trajectory.                                        #
# usage: plot_rmsd_namd(file)                                                       #
#####################################################################################
def plot_rmsd_namd(file):
    from matplotlib import cm,colors
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure, show, rc
    import numpy as np
    import pandas as pd
    df = pd.read_csv(file)
    fig = plt.figure(figsize=(10,8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()
    # df['Time'] = df['Time']/1000
    b3, = plt.plot(df['frames']/2,df['side']/10,linewidth=2, label="side_atoms")
    b1, = plt.plot(df['frames']/2,df['all']/10,linewidth=2, label="all_atoms")
    b2, = plt.plot(df['frames']/2,df['main']/10,linewidth=2, label="main_atoms")
    b4, = plt.plot(df['frames']/2,df['pro']/10,linewidth=2, label="protein")
    
    plt.xlabel('Time/(ns)', fontproperties="Times New Roman",fontsize=15,weight="bold")
    plt.ylabel('RMSD/(nm)', fontproperties="Times New Roman",fontsize=15,weight="bold")
    # plt.ylabel('Frequency',fontproperties="Times New Roman",fontsize=28,weight="bold")   # 设置y轴标签
    plt.xticks(font="Times New Roman",rotation=0,size=12)      # size must be after the font.
    plt.yticks(font="Times New Roman",size=12)
    # plt.title('Frequency_vdw', fontproperties='Times New Roman', fontsize=33)   # 设置图片标题
    plt.legend(handles=[b1,b2,b3,b4],loc=(0.46,0.84),ncol=2,frameon=False,prop="Times New Roman")    #显示图例，loc图例显示位置(可以用坐标方法显示），ncol图例显示几列，默认为1列,frameon设置图形边框
    # plt.legend(handles=[b1,b2,b4],loc=(0.46,0.84),ncol=2,frameon=False,prop="Times New Roman")
    plt.ylim(0, 2)
    # plt.ylim(0, 10)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, weight="bold")
    plt.show()
    fig.savefig('huitu.pdf')

#####################################################################################
# Generate .pdb for NAMD using the H++ structure.                                   #
# usage: generate_h_pdb(file)                                                       #
#####################################################################################
def generate_h_pdb(file):
    atom_map = {"SER":{"N":"N", "H1":"HT1", "H2":"HT2", "H3":"HT3", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "OG":"OG", "HG":"HG1", "C":"C", "O":"O", "H":"HN"}, 
        "GLY":{"N":"N", "H":"HN", "CA":"CA", "HA2":"HA1", "HA3":"HA2", "C":"C", "O":"O"},
        "PHE":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "CD1":"CD1", "HD1":"HD1", "CE1":"CE1", "HE1":"HE1", "CZ":"CZ", "HZ":"HZ", "CE2":"CE2", "HE2":"HE2", "CD2":"CD2", "HD2":"HD2", "C":"C", "O":"O"},
        "ARG":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "CD":"CD", "HD2":"HD1", "HD3":"HD2", "NE":"NE", "HE":"HE", "CZ":"CZ", "NH1":"NH1", "HH11":"HH11", "HH12":"HH12", "NH2":"NH2", "HH21":"HH21", "HH22":"HH22", "C":"C", "O":"O"},
        "LYS":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "CD":"CD", "HD2":"HD1", "HD3":"HD2", "CE":"CE", "HE2":"HE1", "HE3":"HE2", "NZ":"NZ", "HZ1":"HZ1", "HZ2":"HZ2", "HZ3":"HZ3", "C":"C", "O":"O"},
        "MET":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "SD":"SD", "CE":"CE", "HE1":"HE1", "HE2":"HE2", "HE3":"HE3", "C":"C", "O":"O"},
        "ALA":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB1":"HB1", "HB2":"HB2", "HB3":"HB3", "C":"C", "O":"O"},
        "PRO":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "CD":"CD", "HD2":"HD1", "HD3":"HD2", "C":"C", "O":"O"},
        "VAL":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB":"HB", "CG1":"CG1", "HG11":"HG11", "HG12":"HG12", "HG13":"HG13", "CG2":"CG2", "HG21":"HG21", "HG22":"HG22", "HG23":"HG23", "C":"C", "O":"O"},
        "GLU":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "CD":"CD", "OE1":"OE1", "OE2":"OE2", "C":"C", "O":"O"},
        "CYS":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "SG":"SG", "HG":"HG1", "C":"C", "O":"O"},
        "GLN":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG2":"HG1", "HG3":"HG2", "CD":"CD", "OE1":"OE1", "NE2":"NE2", "HE21":"HE21", "HE22":"HE22", "C":"C", "O":"O"},
        "THR":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB":"HB", "CG2":"CG2", "HG21":"HG21", "HG22":"HG22", "HG23":"HG23", "OG1":"OG1", "HG1":"HG1", "C":"C", "O":"O"},
        "LEU":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "HG":"HG", "CD1":"CD1", "HD11":"HD11", "HD12":"HD12", "HD13":"HD13", "CD2":"CD2", "HD21":"HD21", "HD22":"HD22", "HD23":"HD23", "C":"C", "O":"O"},
        "ASN":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "OD1":"OD1", "ND2":"ND2", "HD21":"HD21", "HD22":"HD22", "C":"C", "O":"O"},
        "TRP":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "CD1":"CD1", "HD1":"HD1", "NE1":"NE1", "HE1":"HE1", "CE2":"CE2", "CZ2":"CZ2", "HZ2":"HZ2", "CH2":"CH2", "HH2":"HH2", "CZ3":"CZ3", "HZ3":"HZ3", "CE3":"CE3", "HE3":"HE3", "CD2":"CD2", "C":"C", "O":"O"},
        "ASP":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "OD1":"OD1", "OD2":"OD2", "C":"C", "O":"O"},
        "HIS":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "ND1":"ND1", "CE1":"CE1", "HE1":"HE1", "NE2":"NE2", "HE2":"HE2", "CD2":"CD2", "HD2":"HD2", "C":"C", "O":"O", "HD1":"HD1"},
        "TYR":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB2":"HB1", "HB3":"HB2", "CG":"CG", "CD1":"CD1", "HD1":"HD1", "CE1":"CE1", "HE1":"HE1", "CZ":"CZ", "OH":"OH", "HH":"HH", "CE2":"CE2", "HE2":"HE2", "CD2":"CD2", "HD2":"HD2", "C":"C", "O":"O"},
        "ILE":{"N":"N", "H":"HN", "CA":"CA", "HA":"HA", "CB":"CB", "HB":"HB", "CG2":"CG2", "HG21":"HG21", "HG22":"HG22", "HG23":"HG23", "CG1":"CG1", "HG12":"HG11", "HG13":"HG12", "CD1":"CD1", "HD11":"HD1", "HD12":"HD2", "HD13":"HD3", "C":"C", "O":"O"},
    }
    with open(file) as f:
        f1 = f.readlines()
    result = open("com21-receptor-h-name.pdb", "w")
    for i in f1:
        if i.startswith("ATOM"):
            if i[12:16].strip() in atom_map[i[17:20].strip()].keys():
                print("000")
                result.write(i[0:12] + "{:^4s}".format(atom_map[i[17:20].strip()][i[12:16].strip()]) + i[16:])
            else:
                print("111")
                result.write(i)
        else:
            result.write(i)

#####################################################################################
# Plot clusters-time figure.                                                        #
# usage: plot_clusters_time(file_in)                                                #
#####################################################################################
def plot_clusters_time(file_in):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_in)

    fig=plt.figure(figsize=(9,5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)           #设置绘图区域大小位置

    # plt.scatter(df["frame0"]/2, df["cluster0"]+1, s=40, linewidths=0, edgecolors="k", label="cluster1")
    plt.scatter(df["cluster1"]/2, np.full(shape=(len(df["cluster1"])), fill_value=round(len(df["cluster1"].dropna())/4, 2)), s=100, alpha=1, marker="+", linewidths=1, edgecolors="k", label="cluster1",)
    plt.scatter(df["cluster2"]/2, np.full(shape=(len(df["cluster2"])), fill_value=round(len(df["cluster2"].dropna())/4, 2)), s=100, alpha=0.5, marker="+", linewidths=1, edgecolors="k", label="cluster2",)
    plt.scatter(df["cluster3"]/2, np.full(shape=(len(df["cluster3"])), fill_value=round(len(df["cluster3"].dropna())/4, 2)), s=100, alpha=0.4, marker="+", linewidths=1, edgecolors="k", label="cluster3",)
    plt.scatter(df["cluster4"]/2, np.full(shape=(len(df["cluster4"])), fill_value=round(len(df["cluster4"].dropna())/4, 2)), s=100, alpha=0.3, marker="+", linewidths=1, edgecolors="k", label="cluster4",)
    plt.scatter(df["cluster5"]/2, np.full(shape=(len(df["cluster5"])), fill_value=round(len(df["cluster5"].dropna())/4, 2)), s=100, alpha=0.3, marker="+", linewidths=1, edgecolors="k", label="cluster5",)
    # plt.scatter(df["frame1"]/2, df["cluster1"]+1, s=40, linewidths=0, edgecolors="k", label="cluster2")
    # plt.scatter(df["frame2"]/2, df["cluster2"]+1, s=40, linewidths=0, edgecolors="k", label="cluster3")
    # plt.scatter(df["frame3"]/2, df["cluster3"]+1, s=40, linewidths=0, edgecolors="k", label="cluster4")
    # plt.scatter(df["frame4"]/2, df["cluster4"]+1, s=40, linewidths=0, edgecolors="k", label="cluster5")

    plt.xticks(font="Times New Roman",size=9)
    plt.yticks(font="Times New Roman",size=9)
    # min_ = min(min(df["train_label"]), min(df["val_label"]), min(df["outtest_label"]), min(df["train_pred"]), min(df["val_pred"]), min(df["outtest_pred"])) - 1
    # max_ = max(max(df["train_label"]), max(df["val_label"]), max(df["outtest_label"]), max(df["train_pred"]), max(df["val_pred"]), max(df["outtest_pred"])) + 1
    # m = np.arange(min_, max_, 0.01)
    # plt.plot(m, m, "r--", lw=0.5,)
    # plt.xlim(min_, max_)
    plt.ylim(0, 120)
    plt.xlim(-5, 205)

    plt.legend(loc=(0.55,0.87),ncol=3,frameon=False,prop="Times New Roman")

    plt.xlabel('Time/ns', fontproperties="Times New Roman",fontsize=12,weight="bold")
    plt.ylabel('Clusters Percentage/%', fontproperties="Times New Roman",fontsize=12,weight="bold")

    plt.show()
    fig.savefig("Figure.pdf")
    
#####################################################################################
# Finding the needed structure from library according to SMILES or similarity.      #
# usage: find_structure_smiles(file, smiles)                                        #
#####################################################################################
def smiles_can(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveHs(mol)
    # canonical_smi = Chem.MolToSmiles(mol)
    # return canonical_smi
    return mol

def find_structure_smiles(file_in, smiles):
    from rdkit.Chem import AllChem as Chem
    from rdkit import DataStructs
    with open(file_in) as f:
        f1 = f.readlines()
    info_dict = {}
    for i in range(1, len(f1)):
        smi_tmp = f1[i].split(",")[3]
        info_dict[f1[i]] = smiles_can(smi_tmp)
    smiles_in_can = smiles_can(smiles)
    fp_ref = Chem.GetMorganFingerprintAsBitVect(smiles_in_can, 2)
    result = open("result.csv", "w")
    for i in info_dict.keys():
        # if smiles_in_can == info_dict[i]:
        fp = Chem.GetMorganFingerprintAsBitVect(info_dict[i], 2)
        score = DataStructs.DiceSimilarity(fp, fp_ref)
        result.write(i.replace("\n", ",") + str(score) + "\n")

#####################################################################################
# Process PBC.                                                                      #
# usage: process_pbc()                                                              #
# Align the trajectory of NAMD.                                                     #
# usage: align_dcd()                                                                #
# Clustering the trajectory of NAMD.                                                #
# usage: cluster_dcd()                                                              #
# Find the centroid of a cluster.                                                   #
# usage: find_centroid()                                                            #
#####################################################################################
def process_pbc(psf, trajectory):
    import MDAnalysis as mda
    import MDAnalysis.transformations as trans

    traj = mda.Universe(psf, trajectory)
    # print(traj.select_atoms())
    # traj = mda.coordinates.DCD.DCDReader(trajectory, convert_units=True) 
    # with traj.Writer("test.dcd") as result:
    #     for ts in traj.trajectory:
    #         result.write(traj)
    # traj.Writer("test.dcd")
    # traj = mda.Universe(psf, traj)
    print(traj)
    pro_lig = traj.select_atoms("segid PRO or (segid LIG)")
    not_pro_lig = traj.select_atoms("not (segid PRO or (segid LIG))")
    all_ = traj.select_atoms("all")
    transforms = [trans.unwrap(pro_lig),
              trans.center_in_box(pro_lig, wrap=True),
              trans.wrap(not_pro_lig)]
    traj.trajectory.add_transformations(*transforms)
    # with mda.Writer("test.dcd", multiframe=True) as result:
    with mda.Writer("test.dcd", n_atoms=all_.n_atoms) as result:
        for ts in traj.trajectory:
            result.write(traj)

def align_dcd():
    pass

def find_centroid(trajectory, topology):
    import mdtraj as md
    import numpy as np
    import pandas as pd

    df = pd.read_csv("huitu_in.csv")
    cluster_frames = np.array(df["frame0"].dropna(), dtype=np.int32)-1
    cluster_frames = cluster_frames[cluster_frames >= 0]
    # cluster_frames = [i for i in cluster_frames if i != "nan"]
    # print(cluster_frames)
    traj = md.load(trajectory, top=topology)
    # print(traj)
    traj_cluster = md.join(traj[i] for i in cluster_frames)
    atom_indices = [a.index for a in traj.topology.atoms if ((a.residue.name == 'LIG') and (a.element.symbol != 'H'))]
    distances = np.empty((traj_cluster.n_frames, traj_cluster.n_frames))
    for i in range(traj_cluster.n_frames):
        distances[i] = md.rmsd(traj_cluster, traj_cluster, i, atom_indices=atom_indices)
    index = np.exp(-1*distances / distances.std()).sum(axis=1).argmax()
    print(index, cluster_frames[index])
    # traj_cluster[index].save_pdb("cluster5_centroid.pdb")
    traj_cluster[index].save("cluster1_centroid.pdb")
    traj_cluster[index].save("cluster1_centroid.dcd")

def find_centroid_v2(clusters, rmsd_matrix):
    import numpy as np
    import pandas as pd

    with open(rmsd_matrix) as f:
        f1 = f.readlines()
    matrix = []
    for i in f1:
        matrix.append(i.strip("\n,").split(","))
    # matrix = np.matrix(matrix, dtype=np.float64)
    # print(matrix)

    df = pd.read_csv(clusters)
    result = open("centroid.dat", "w")
    for i in range(1, 6):
        cluster_frames = np.array(df["cluster" + str(i)].dropna(), dtype=np.int32)
        cluster_matrix_row = [matrix[i] for i in cluster_frames]
        # print(len(cluster_matrix_row))
        cluster_matrix_col = []
        for a in cluster_matrix_row:
            temp = []
            for b in cluster_frames:
                temp.append(a[b])
            cluster_matrix_col.append(temp)
        cluster_matrix_col = np.matrix(cluster_matrix_col, dtype=np.float64)
        # print(cluster_matrix_col.shape)
        index = np.exp(-1*cluster_matrix_col / cluster_matrix_col.std()).sum(axis=1).argmax()
        result.write(str(i) + ":" + str(cluster_frames[index]) + "\n")

#####################################################################################
# Clustering according to rmsd similarity.                                          #
# usage: cluster_rmsd(file_in)                                                      #
#####################################################################################
def cluster_rmsd(file_in):
    from scipy.cluster.hierarchy import centroid, fcluster, ward, maxRstat, fclusterdata
    from scipy.spatial.distance import pdist
    import numpy as np

    with open(file_in) as f:
        f1 = f.readlines()
    matrix = []
    for i in f1:
        matrix.append(i.strip("\n,").split(","))
    matrix = np.matrix(matrix, dtype=np.float64)
    y = pdist(matrix)
    print(max(y))
    Z = ward(y)
    print(Z)
    # clusters = fcluster(Z, 1.1, criterion='monocrit', monocrit=MR)
    clusters = fcluster(Z, 100, criterion='distance')
    print(clusters)

#####################################################################################
# Set the simulation water box in NAMD.                                             #
# usage: set_watbox(file_in)                                                        #
#####################################################################################
def set_watbox(file_in):
    with open(file_in) as f:
        f1 = f.readlines()
    x = []
    y = []
    z = []
    for i in f1:
        if i.startswith("ATOM"):
            x.append(float(i[30:38]))
            y.append(float(i[38:46]))
            z.append(float(i[46:54]))
    center = ((max(x)+min(x))/2, (max(y)+min(y))/2, (max(z)+min(z))/2)
    x_com = max(x) - min(x)
    y_com = max(y) - min(y)
    z_com = max(z) - min(z)
    radius = max(x_com, y_com, z_com)/2 + 5
    x_min = int(center[0] - radius)
    x_max = int(center[0] + radius)
    y_min = int(center[1] - radius)
    y_max = int(center[1] + radius)
    z_min = int(center[2] - radius)
    z_max = int(center[2] + radius)
    print("{{{{{0} {1} {2}}} {{{3} {4} {5}}}}}".format(x_min, y_min, z_min, x_max, y_max, z_max))
    return x_min, x_max, y_min, y_max, z_min, z_max

#####################################################################################
# Find the arrounding residues of ligand according the centroid structure.          #
# usage: find_arround_resi(file_in)                                                 #
# Align and RMSD calculation.                                                       #
# usage: align_rmsd()                                                               #
# 
#####################################################################################
# def find_arround_resi(file_in):
def find_arround_resi():
    from pymol import cmd

    cmd.delete("all")
    # cmd.load(file_in)
    cmd.load("cluster1_centroid.pdb")
    myspace = {"lst":[]}
    cmd.iterate("(br. all within 10 of resn LIG) and (not resn LIG) and polymer.protein and (name CA)", "lst.append(resi)", space = myspace)
    # lst_ = [int(i) for i in myspace["lst"]]
    # lst_ = sorted(cmd.identify("(br. all within 10 of resn LIG) and (not resn LIG) and polymer.protein and (not hydrogens)", 0))
    # print(lst_)
    a = [int(i) for i in myspace["lst"]]
    # a = myspace["lst"]
    # print(a)
    all_lst = []
    s = []  # 定义一个空列表
    for i in a:
        if len(s) == 0 or s[-1] + 1 == i:  # 判断，如果是空列表或者列表中最后一个值加1等于当前值
            s.append(i)  # 写入列表
        else:  # 不满足以上条件
            # if len(s) >= 2:  # 判断长度是否大于2， 大于2则打印出来
            all_lst.append(s)
            s = [i]  # 给s重新复制为i
    # if len(s) >= 2:  # 最后一轮，判断是否是连续值，是的话就打印
    all_lst.append(s)
    all_resid = []
    for i in all_lst:
        if len(i) == 1:
            all_resid.append("(resid "+str(i[0])+")")
        else:
            all_resid.append("(resid "+str(i[0])+" to "+str(i[-1])+")")
    print(" or ".join(all_resid))
    # return lst_

def align_rmsd(trajectory, topology, lst):
    import mdtraj as md

    traj = md.load(trajectory, top=topology)
    # print(traj)
    # atom_indices = [a.index for a in traj.topology.atoms if ((a.residue.index == 9999) and (a.element.symbol != 'H'))]
    # atom_indices = [a.index for a in traj.topology.atoms if ((a.residue.index in lst) and (a.element.symbol != 'H'))]
    atom_indices_pocket = [i-1 for i in lst]
    # print(atom_indices_pocket)
    # print(traj.topology.select_expression("resi 1 to 2"))
    traj_align = md.Trajectory.superpose(traj, traj, atom_indices=atom_indices_pocket)
    # traj_align.save("align_pocket.dcd")
    pocket_rmsd = md.rmsd(traj_align, traj_align, 0, atom_indices=atom_indices_pocket)
    atom_indices_all_lig = [a.index for a in traj.topology.atoms if ((a.residue.name == "LIG") and (a.element.symbol != 'H'))]
    # print(atom_indices_all_lig)
    all_lig_rmsd = md.rmsd(traj_align, traj_align, 0, atom_indices=atom_indices_all_lig, precentered=True)
    print(all_lig_rmsd)
    atom_indices_main_lig = [a.index for a in traj.topology.atoms if ((a.residue.name == "LIG") and (a.name in ["C8", "N4", "C9", "C10", "C14", "C13", "C12", "C11", "C7", "C15", "N3", "C6", "O2", "C5", "C16", "C17", "C18", "CL", "C19", "C20", "C21"]))]
    # print(atom_indices_main_lig)
    main_lig_rmsd = md.rmsd(traj_align, traj_align, 0, atom_indices=atom_indices_main_lig)
    atom_indices_side_lig = [a.index for a in traj.topology.atoms if ((a.residue.name == "LIG") and (a.name in ["C22", "O3", "N2", "C4", "C3", "C2", "O1", "N1", "C1"]))]
    # print(atom_indices_side_lig)
    side_lig_rmsd = md.rmsd(traj_align, traj_align, 0, atom_indices=atom_indices_side_lig)
    print(md.rmsd(traj_align, traj_align, 2, precentered=True))
    atom_indices_protein_backbone = [a.index for a in traj.topology.atoms if (a.residue.is_protein and a.is_backbone)]
    # print(traj.topology.select_expression("protein and backbone"))
    print(md.rmsd(traj, traj, 0, atom_indices=atom_indices_protein_backbone))

    print(md.rmsd(traj_align, traj_align, 0, atom_indices=atom_indices_all_lig,)[-10:])
    print(md.rmsd(traj, traj, 0, atom_indices=atom_indices_all_lig,)[-10:])

#####################################################################################
# Calculate ddg.                                                                    #
# usage: calculate_ddg()                                                            #
#####################################################################################
def calculate_ddg(bonded, free):
    import numpy as np
    import math

    complex_fep = []
    with open(bonded) as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        complex_fep.append(f1[i].strip().split(",")[1])
    RNA_fep = []
    with open(free) as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        RNA_fep.append(f1[i].strip().split(",")[1])

    complex_fep = np.array(complex_fep, dtype=np.float64)
    RNA_fep = np.array(RNA_fep, dtype=np.float64)
    complex_mean = np.mean(complex_fep)
    RNA_mean = np.mean(RNA_fep)
    complex_se = np.std(complex_fep, ddof=1)/math.sqrt(len(complex_fep))
    RNA_se = np.std(RNA_fep, ddof=1)/math.sqrt(len(RNA_fep))
    ddg_mean = complex_mean - RNA_mean
    ddg_se = math.sqrt(complex_se*complex_se + RNA_se*RNA_se)
    print("The mean of ddg (kJ/mol):", ddg_mean)
    print("The SE of ddg (kJ/mol):", ddg_se)
    # print("The mean of ddg (kcal/mol):", ddg_mean/4.184)
    # print("The SE of ddg (kcal/mol):", ddg_se/4.184)
    result = open("ddg_result.txt", "w")
    # result.write("The ddg (kJ/mol): " + str(ddg_mean) + " +/- " + str(ddg_se) + "\n")
    result.write("The ddg (kcal/mol): " + str(ddg_mean) + " +/- " + str(ddg_se) + "\n")

#####################################################################################
# Find structure containing specified substructure using rdkit.                     #
# usage: ()                                                            #
#####################################################################################

#####################################################################################
# plot histogram.                     #
# usage: ()                                                            #
#####################################################################################
def plot_histogram(file_in):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_in)

    x_ = df["activity"]
    y_ = df["counts"]
   
    fig=plt.figure(figsize=(10,6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    p1 = plt.bar(x_, y_, width=0.3, label='value', edgecolor='#00AFBB', linewidth=0, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',],)
    plt.bar_label(p1, label_type='edge')

    plt.xticks(ticks=x_, font="Times New Roman",size=9)
    plt.yticks(font="Times New Roman",size=9)

    # plt.xlim(0.5, 29)
    # plt.ylim(0, 41)

    plt.xlabel('IC50/uM', fontproperties="Times New Roman",fontsize=12,weight="bold")
    plt.ylabel('Counts', fontproperties="Times New Roman",fontsize=12,weight="bold")

    plt.show()
    fig.savefig("Figure.pdf")

#####################################################################################
# Similarity RDKit.                                                                 #
# usage: ()                                                             #
#####################################################################################
def MACCS_similarity(file_in):
    import pandas as pd
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys

    df = pd.read_csv(file_in)

    rt = open("MACCS_similarity.csv", "w")
    log = open("run.log", "w")
    ids = list(df["CID"])
    # print(ids[0])
    smiles = list(df["SMILES"])
    activity = list(df["f_avg_IC50"])
    rt.write(","+",".join(ids)+"\n")
    for a in range(len(smiles)):
        log.write(str(a) + "\n")
        temp = ids[a]
        a_smi = Chem.MolFromSmiles(smiles[a])
        fp_a = MACCSkeys.GenMACCSKeys(a_smi)
        for b in range(len(smiles)):
            b_smi = Chem.MolFromSmiles(smiles[b])
            fp_b = MACCSkeys.GenMACCSKeys(b_smi)
            similarity = DataStructs.FingerprintSimilarity(fp_a, fp_b)
            temp = temp + "," + str(round(similarity, 3))
        temp += "\n"
        rt.write(temp)
        log.flush()
    rt.close()
    log.close()

def plot_maccs_similarity_heatmap(file_in):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import palettable

    df=pd.read_csv(file_in)
    df = df[df.columns[1:]]
    # df.index = range(1, df.columns.values+1)
    fig = plt.figure(figsize=(10, 6),dpi=100)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)  
    sns.heatmap(data=df,
        vmin = -0.3,
        # vmax = 60,
        # cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
        # cmap=plt.cm.Spectral,
        cmap=sns.color_palette('RdYlBu',100),
        mask=np.triu(np.ones_like(df,dtype=np.bool)),
        # annot=True,
        # fmt=".2f",
    )
    plt.show()
    fig.savefig("Figure_MACCS_color.pdf")

#####################################################################################
# SMILES can                                                                        #
# usage: ()                                                             #
#####################################################################################
def SMILES_can(file_in):
    import pandas as pd
    from rdkit import Chem

    df = pd.read_csv(file_in)

    rt = open("activity_data-20221215_have_activity_dedup.csv", "w")
    ids = list(df["CID"])
    # print(ids[0])
    smiles = list(df["SMILES"])
    activity = list(df["f_avg_IC50"])
    rt.write("CID,SMILES,f_avg_IC50\n")
    for i in range(len(ids)):
        mol = Chem.MolFromSmiles(smiles[i])
        canonical_smi = Chem.MolToSmiles(mol)
        rt.write(ids[i]+","+canonical_smi+","+str(activity[i])+"\n")
        rt.flush()
    rt.close()

def molvs_can(file_in):
    from molvs import standardize_smiles

    with open(file_in) as f:
        f1 = f.readlines()
    output = ""
    for i in f1:
        name = i.split(",")[0]
        smiles = i.split(",")[1].strip()
        smiles_stand = standardize_smiles(smiles)
        output = output + name + "," + smiles_stand + "\n"
    with open("smiles_standard.txt","w") as rt:
        rt.write(output)

def generate_input():
    import math
    id_activity = {}
    with open("activity_data-20221215_have_activity.csv") as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        id_activity[f1[i].split(",")[0]] = f1[i].split(",")[2].replace("\n", "")

    rt = open("mmp_input.csv", "w")
    rt.write("Source_Mol_ID,Target_Mol_ID,Source_Mol,Target_Mol,Source_Mol_IC50,Target_Mol_IC50,Delta_IC50,Transformation,Core\n")

    with open("mmp_indexed.csv") as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        tmp_lst = f1[i].replace("\n", "").split("\t")
        delta_ic = math.log2(float(id_activity[tmp_lst[3]])/float(id_activity[tmp_lst[2]]))
        rt.write(tmp_lst[2]+","+tmp_lst[3]+","+tmp_lst[0]+","+tmp_lst[1]+","+id_activity[tmp_lst[2]]+","+id_activity[tmp_lst[3]]+","+str(delta_ic)+","+tmp_lst[4]+","+tmp_lst[5]+"\n")

#####################################################################################
# desol                                                                             #
# usage: ()                                                             #
#####################################################################################
def desol():
    from pymol import cmd

    with open("id") as f:
        f1 = f.readlines()

    for i in f1:
        id = i.strip()
        cmd.load("cluster" + id + "_centroid.pdb")
        cmd.remove("sol")
        cmd.remove("inorganic")
        cmd.remove("resn CLA")
        cmd.save("cluster" + id + "_centroid_desol.pdb")
        cmd.delete("all")

#####################################################################################
# MACCS Similarity with ZINC.                                                       #
# usage: similarity_zinc()                                                          #
#####################################################################################
def similarity_zinc():
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys

    rt = open("I3C-3_similarity.txt", "w")
    log = open("run.log", "w")
    i = 0
    ref_smi = Chem.MolFromSmiles("CNC(=O)CN1C[C@@H](C(=O)Nc2cncc3ccccc23)c2cc(Cl)ccc2C1=O")
    fp_ref = MACCSkeys.GenMACCSKeys(ref_smi)
    with open("../ZINC/ZINC_all.smi") as f:
        for a in f:
            a_smi = Chem.MolFromSmiles(a.split()[0])
            fp_a = MACCSkeys.GenMACCSKeys(a_smi)
            similarity = DataStructs.FingerprintSimilarity(fp_a, fp_ref)
            if similarity >= 0.75:
                rt.write(a.replace("\n", "")+" "+str(round(similarity, 3))+"\n")
            i += 1
            if i%100000 == 0:
                log.write(str(i) + "\n")
            rt.flush()    
            log.flush()
    rt.close()
    log.close()

def similarity_zinc_topol():
    from rdkit import Chem
    from rdkit import DataStructs

    rt = open("I3C-3_similarity_topol.txt", "w")
    log = open("run_topol.log", "w")
    i = 0
    ref_smi = Chem.MolFromSmiles("CNC(=O)CN1C[C@@H](C(=O)Nc2cncc3ccccc23)c2cc(Cl)ccc2C1=O")
    fp_ref = Chem.RDKFingerprint(ref_smi)
    with open("../ZINC/ZINC_all.smi") as f:
        for a in f:
            a_smi = Chem.MolFromSmiles(a.split()[0])
            fp_a = Chem.RDKFingerprint(a_smi)
            similarity = DataStructs.FingerprintSimilarity(fp_a, fp_ref)
            if similarity >= 0.75:
                rt.write(a.replace("\n", "")+" "+str(round(similarity, 3))+"\n")
            i += 1
            if i%100000 == 0:
                log.write(str(i) + "\n")
            rt.flush()    
            log.flush()
    rt.close()
    log.close()

def similarity_3cl_inhouse():
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys

    rt = open("I3C-3_similarity_inhouse_maccs.csv", "w")
    rt.write("CID,SMILES,f_avg_IC50,Similarity\n")
    ref_smi = Chem.MolFromSmiles("CNC(=O)CN1C[C@@H](C(=O)Nc2cncc3ccccc23)c2cc(Cl)ccc2C1=O")
    fp_ref = MACCSkeys.GenMACCSKeys(ref_smi)
    with open("activity_data-20221215_have_activity.csv") as f:
        f1 = f.readlines()
        for a in range(1, len(f1)):
            a_smi = Chem.MolFromSmiles(f1[a].split(",")[1])
            fp_a = MACCSkeys.GenMACCSKeys(a_smi)
            similarity = DataStructs.FingerprintSimilarity(fp_a, fp_ref)
            rt.write(f1[a].replace("\n", "")+","+str(round(similarity, 3))+"\n")
            rt.flush()    
    rt.close()

def similarity_inhouse_topol():
    from rdkit import Chem
    from rdkit import DataStructs

    rt = open("I3C-3_similarity_inhouse_topol.csv", "w")
    rt.write("CID,SMILES,f_avg_IC50,Similarity\n")
    ref_smi = Chem.MolFromSmiles("CNC(=O)CN1C[C@@H](C(=O)Nc2cncc3ccccc23)c2cc(Cl)ccc2C1=O")
    fp_ref = Chem.RDKFingerprint(ref_smi)
    with open("activity_data-20221215_have_activity.csv") as f:
        f1 = f.readlines()
        for a in range(1, len(f1)):
            a_smi = Chem.MolFromSmiles(f1[a].split(",")[1])
            fp_a = Chem.RDKFingerprint(a_smi)
            similarity = DataStructs.FingerprintSimilarity(fp_a, fp_ref)
            rt.write(f1[a].replace("\n", "")+","+str(round(similarity, 3))+"\n")
            rt.flush()    
    rt.close()

def LigDream_gen():
    from compound_generation import CompoundGenerator
    from rdkit.Chem import AllChem as Chem
    import os

    my_gen = CompoundGenerator(use_cuda=True)

    vae_weights =  os.path.join(".", "modelweights/vae-210000.pkl")
    encoder_weights =  os.path.join(".", "modelweights/encoder-210000.pkl")
    decoder_weights =os.path.join(".", "modelweights/decoder-210000.pkl")

    my_gen.load_weight(vae_weights, encoder_weights, decoder_weights)

    with open("I3C-3_similarity_inhouse_topol_picked.csv") as f:
        f1 = f.readlines()
    rt = open("I3C-3_ligdream_gen.csv", "w")
    for i in range(1, len(f1)):
        seed_mol = f1[i].split(",")[0]
        seed_mol_name = f1[i].split(",")[1]
        seed_mol_activity = f1[i].split(",")[3]
        rt.write(seed_mol_name + "," + seed_mol + "," + seed_mol_activity + ",")

        tmp = []
        for a in range(5):
            gen_mols = my_gen.generate_molecules(seed_mol,
                                                n_attemps=100,  # How many attemps of generations will be carried out
                                                lam_fact=1.,  # Variability factor
                                                probab=False,  # Probabilistic RNN decoding
                                                filter_unique_valid=True)  # Filter out invalids and replicates
            for i in gen_mols:
                tmp.append(Chem.MolToSmiles(i))
        
        rt.write(",".join(tmp) + "\n")
        rt.flush()

def de_fepout():
    import os

    with open("complex-prod-forward.fepout") as f:
        f1 = f.readlines()
    rt = open("complex-prod-forward.fepouttmp", "w")
    for i in f1:
        if i.startswith("FepEnergy:"):
            step = int(i.split()[1])/5
            if step%10 == 0:
                rt.write(i)
        else:
            rt.write(i)
    rt.close()
    os.remove("complex-prod-forward.fepout")
    os.rename("complex-prod-forward.fepouttmp", "complex-prod-forward.fepout")

def process_de_fepout(in_file):
    import os

    os.chdir(os.path.join(in_file, "bonded", "simulation", "prod"))
    for i in range(1, 7):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("../../../")

    os.chdir(os.path.join("free", "simulation", "prod"))
    for i in range(1, 7):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("../../../")

    os.chdir(os.path.join("free_1", "simulation", "prod"))
    for i in range(1, 7):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("../../../")

    os.chdir(os.path.join("bonded_1", "simulation", "prod"))
    for i in range(1, 7):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("../../../")

def process_de_fepout_v2(in_file):
    import os

    os.chdir(os.path.join(in_file, "bonded_final"))
    for i in range(1, 6):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("..")

    os.chdir(os.path.join("free_final"))
    for i in range(1, 6):
        os.chdir(os.path.join("complex" + str(i)))
        de_fepout()
        os.chdir("..")
    os.chdir("..")

def get_fragment_zinc(in_file):
    from rdkit import Chem
    from rdkit.Chem import BRICS
    import dbm
    import struct

    outfname = in_file + '.dbm'
    outlog = in_file + '.log'
    db =  dbm.open(outfname, flag='n')
    rt = open(outlog, "w")
    idx = 0
    with open(in_file) as f:
        for smi in f:
            idx += 1
            if len(smi.split()) == 2:
                m = Chem.MolFromSmiles(smi.split()[0])
                if m is None or m.GetNumHeavyAtoms()>60: continue
                s = BRICS.BRICSDecompose(m)
                for entry in s:
                    cnt = struct.unpack('I', db.get(entry,b'\0\0\0\0'))[0]+1
                    db[entry] = struct.pack('I',cnt)
                if idx%1000 == 0:
                    rt.write(str(idx)+"\n")
                    rt.flush()
            else:
                pass
    db.close()

def get_smis():
    all_f = []
    for i in range(100):
        all_f.append("chembl_"+str(i).zfill(2))
    return all_f

#####################################################################################
# run the program                                                                   #
# usage: you can pick some functions to run just like playing Lego !!!              #
#####################################################################################
def main():
    # import sys

    # find_centroid(sys.argv[1], sys.argv[2])
    # plot_rmsd_namd(sys.argv[1])
    # plot_clusters_time(sys.argv[1])
    # plot_histogram(sys.argv[1])
    # process_pbc(sys.argv[1], sys.argv[2])
    # find_arround_resi()
    # # align_rmsd(sys.argv[1], sys.argv[2], lst)
    # # cluster_rmsd(sys.argv[1])
    # find_centroid_v2(sys.argv[1], sys.argv[2])
    # calculate_ddg(sys.argv[1], sys.argv[2])
    # MACCS_similarity(sys.argv[1])
    # plot_maccs_similarity_heatmap(sys.argv[1])
    # SMILES_can(sys.argv[1])
    # molvs_can(sys.argv[1])
    # generate_input()
    # desol()
    # similarity_zinc()
    # similarity_zinc_topol()
    # similarity_3cl_inhouse()
    # similarity_inhouse_topol()
    # LigDream_gen()
    # de_fepout()
    # process_de_fepout(sys.argv[1])
    # process_de_fepout_v2(sys.argv[1])

    from multiprocessing import Pool
    
    all_f = get_smis()
    with Pool(128) as p:
        p.map(get_fragment_zinc, all_f)

if __name__=="__main__":
    main() 
