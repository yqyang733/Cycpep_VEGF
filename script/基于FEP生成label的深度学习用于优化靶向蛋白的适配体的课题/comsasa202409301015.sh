from pymol import cmd

# 定义选择
PROTEIN_SEL = "polymer.protein"
NUCLEIC_SEL = "polymer.nucleic"
# NUCLEIC_BASES = "resn A+T+G+C+U"  # 核酸碱基

PDB_FILE = "4hdu.ent.pdb"  # 你的PDB文件路径
OUTPUT_FILE = "nucleic_buried_area_per_chain_with_base.txt"

# 加载PDB文件
cmd.load(PDB_FILE)

# 设置探针半径（通常为1.4 Å）
probe_radius = 1.4

# 获取所有核酸链信息
nucleic_chains = cmd.get_chains(NUCLEIC_SEL)

# 打开输出文件
with open(OUTPUT_FILE, "w") as output:
    output.write("Chain\tResidue\tBase\tSASA_PRO\tSASA_base\tSASA_complex\tBuried_SASA\tPercent_Buried\n")

    # 计算蛋白的总SASA
    cmd.create("prot", PROTEIN_SEL)
    sasa_pro = cmd.get_area(selection="prot", load_b=1)   
    cmd.delete("prot")        

    # 遍历每条核酸链
    for chain in nucleic_chains:
        chain_sel = f"{NUCLEIC_SEL} and chain {chain}"

        # 获取该链中的所有残基编号和碱基名称
        residues = list()
        cmd.iterate(f"{chain_sel}", "residues.append((chain, resi, resn))", space=locals())
        residues = set(residues)
        residues = sorted(residues, key=lambda x: (x[0], x[1]))

        # 遍历该链上的每个碱基
        for chain, resid, base_name in residues:
            base_sel = f"{NUCLEIC_SEL} and chain {chain} and resi {resid} and (not backbone)"
            
            # 计算碱基的总SASA
            cmd.create("base", base_sel)
            sasa_base = cmd.get_area(selection="base", load_b=1)
            cmd.delete("base")
            
            # 计算该碱基与蛋白质结合时的SASA
            # combined_sel = f"({PROTEIN_SEL}) or ({base_sel})"
            # 将蛋白和改碱基保存为一个新的名称为alpha的object。
            pro_base_sel = f"{PROTEIN_SEL} or {base_sel}"
            cmd.create("alpha", pro_base_sel)
            combined_sel = f"alpha"
            sasa_complex = cmd.get_area(selection=combined_sel, load_b=1)
            
            # 计算埋藏面积
            buried_sasa = (sasa_pro + sasa_base - sasa_complex)/2
            percent_buried = buried_sasa / sasa_base * 100

            # 写入输出文件，增加碱基名称
            output.write(f"{chain}\t{resid}\t{base_name}\t{sasa_pro:.2f}\t{sasa_base:.2f}\t{sasa_complex:.2f}\t{buried_sasa:.2f}\t{percent_buried:.2f}%\n")

            cmd.delete("alpha")

cmd.quit()
