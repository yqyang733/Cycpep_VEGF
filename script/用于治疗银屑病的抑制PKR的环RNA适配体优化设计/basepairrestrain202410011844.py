import numpy as np
import sys

atom_pair = {"AU":("N1-N3","N6-O4"),"UA":("N3-N1","O4-N6"),"GC":("N2-O2","N1-N3","O6-N4"),"CG":("O2-N2","N3-N1","N4-O6"),"GU":("O6-O4","N1-N3","N2-O2"),"UG":("O4-O6","N3-N1","O2-N2")}

# 读取gro文件
def read_gro_file(gro_file):
    with open(gro_file, 'r') as f:
        lines = f.readlines()
    
    atom_positions = {}
    for line in lines[2:-1]:  # 跳过前两行和最后一行
        resid = line[:5].strip()
        resn = line[5:10].strip()
        atomname = line[10:15].strip()
        index = line[15:20].strip()
        atom_x = line[20:28].strip()
        atom_y = line[28:36].strip()
        atom_z = line[36:44].strip()
        position = np.array([float(atom_x), float(atom_y), float(atom_z)])  # x, y, z坐标
        atom_positions[(resn+resid,atomname)] = (index, position)  # 存储原子名称与索引的对应关系
    
    return atom_positions

# 读取碱基对文件
def read_base_pairs(base_pair_file):
    with open(base_pair_file, 'r') as f:
        lines = f.readlines()
    
    base_pairs = []
    for line in lines:
        base1, base2 = line.strip().split(',')
        base_pairs.append((base1, base2))
    
    return base_pairs

def main():
    gro_file = sys.argv[1]  # .gro 文件
    base_pair_file = sys.argv[2]  # 碱基对文件
    
    # 读取数据
    atom_positions = read_gro_file(gro_file)
    base_pairs = read_base_pairs(base_pair_file)

    output_file = "intermolecular_interactions.dat"
    rt = open(output_file, 'w')
    rt.write("[ intermolecular_interactions]\n")
    rt.write("[ bonds ]\n")
    rt.write("; ai     aj    type   bA      kA   bB      kB\n")

    for i in base_pairs:
        print(i)
        type_pair = i[0][0]+i[1][0]
        for a in atom_pair[type_pair]:
            ai = atom_positions[(i[0],a.split("-")[0])][0]
            ai_pos = atom_positions[(i[0],a.split("-")[0])][1]
            aj = atom_positions[(i[1],a.split("-")[1])][0]
            aj_pos = atom_positions[(i[1],a.split("-")[1])][1]
            distance = np.round(np.linalg.norm(ai_pos - aj_pos),3)
            ai_atm = a.split("-")[0]
            aj_atm = a.split("-")[1]
            if distance <= 0.35:
                rt.write(f"{ai}   {aj}   6   {distance:.3f}   {1000:.1f} ; {i[0]}-{ai_atm} ({ai}) - {i[1]}-{aj_atm} ({aj})\n")
            else:
                # rt.write(f"{ai}   {aj}   6   0.350   {1000:.1f} ; {i[0]}-{ai_atm} ({ai}) - {i[1]}-{aj_atm} ({aj})\n")
                rt.write(f"{ai}   {aj}   6   {distance:.3f}   {1000:.1f} ; {i[0]}-{ai_atm} ({ai}) - {i[1]}-{aj_atm} ({aj})\n")
    rt.close()

if __name__ == "__main__":
    main()
