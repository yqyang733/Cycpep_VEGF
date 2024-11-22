echo 2 |gmx pdb2gmx -f complex.pdb -o build.gro -water tip3p -ignh -missing
python modifyiptfiles.py
sed -i "s/posre_Protein_chain_M.itp/posre_Protein_chain_M_1.itp/g" topol_Protein_chain_M_1.itp  # 这里需要修改itp文件里面的位置限制文件的名称。
sed -i "s/topol_Protein_chain_M.itp/topol_Protein_chain_M_1.itp/g" topol.top
# 并且将build.gro文件中的多余的原子删除。删掉14位K的N上面两个多余的H。
gmx editconf -f build.gro -o newbox.gro -bt cubic -d 0.8
gmx solvate -cp newbox.gro -cs spc216.gro -p topol.top -o solv.gro
gmx grompp -f ~/file/gmx_file/ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo 13|gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname SOD -nname CLA -neutral -conc 0.15
echo q|gmx make_ndx -f solv_ions.gro -o index.ndx
