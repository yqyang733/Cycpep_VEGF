cp ../../MD_rep1/top* -r .
echo 1 0|gmx trjconv -f ../../MD_rep1/prod/step5_1.gro -s ../../MD_rep1/prod/step5_1.tpr -o step5_1.gro -n ../../MD_rep1/index.ndx -center -pbc mol
mkdir mut
cd mut
echo 1 0|gmx trjconv -f ../step5_1.gro -s ../../../MD_rep1/prod/step5_1.tpr -o protein.pdb -n ../../../MD_rep1/index.ndx
export PATH="/public/home/yqyang/software/Miniconda3/envs/pymol/bin:$PATH"
cat > mutation.txt << EOF
52:ARG
63:ARG
153:ARG
EOF
cat > mut.py << EOF
import sys
from pymol import cmd

def residue_mutation(protein, parm):
    with open(parm) as f:
        f1 = f.readlines()

    cmd.load(protein, "pro_mutation")
    cmd.wizard("mutagenesis")
    cmd.do("refresh_wizard")

    for line in f1:
        index = int(line.split(":")[0])
        mutation = line.strip().split(":")[1]
        cmd.get_wizard().set_mode(mutation)
        cmd.get_wizard().do_select("/pro_mutation///%d" % (index))
        cmd.frame(1)
        cmd.get_wizard().apply()
    cmd.save("mut.pdb",'pro_mutation')
    cmd.delete("all")

def main():
    protein = str(sys.argv[1])
    parm = str(sys.argv[2])
    residue_mutation(protein, parm)

if __name__=="__main__":
    main()
EOF
python mut.py protein.pdb mutation.txt
echo 2|gmx pdb2gmx -f mut.pdb -o mut.gro -water tip3p -ignh
cd ../toppar/
mv PROA.itp PROA_1.itp
cp ../mut/topol.top PROA.itp
cp ../mut/posre.itp .
sed -i '31s/.*/PROA             3/' PROA.itp
sed -i '1,27d' PROA.itp
tac PROA.itp | sed '1,20d' | tac > PROA.itptmp
mv PROA.itptmp PROA.itp
echo "" >> AR.itp
echo "#ifdef POSRES_AR" >> AR.itp
echo "[ position_restraints ]" >> AR.itp
echo "  1     1    500    500    500" >> AR.itp
echo "#endif" >> AR.itp
cd ..
sed '1,2d' ./mut/mut.gro > mut_1.gro
tac mut_1.gro | sed '1d' | tac > mut_1.grotmp
mv mut_1.grotmp mut_1.gro
sed '1,2275d' step5_1.gro > step5_1_1.gro
echo "Title" >> step3_input.gro
echo "xxxxx" >> step3_input.gro
cat mut_1.gro >> step3_input.gro
cat step5_1_1.gro >> step3_input.gro
all_line=wc -l step3_input.gro|awk '{print $1}'
atoms=$((${all_line} - 3))
sed -i "2s/.*/${atoms}/" step3_input.gro
rm mut_1.gro step5_1_1.gro
cp -r ../52I63I153I2GGG/mdp/ .
cp -r ../52I63I153I2GGG/job.sh .
gmx make_ndx -f step3_input.gro -o index.ndx << EOF
    1|16
    name 17 SOLU
    13|14|15
    name 18 SOLV
    q
EOF    