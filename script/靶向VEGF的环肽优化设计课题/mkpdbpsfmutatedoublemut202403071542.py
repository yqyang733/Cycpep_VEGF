import os
import sys
import time
import shutil

three_one = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'}
#three_one_need ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 'ASP':'D', 'ASN':'N', 'TRP':'W', 'PHE':'F', 'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'}
three_one_need ={'HSD':'H',}

one_three_need = dict([[v, k] for k, v in three_one_need.items()])

cyclic_pep = {"3":"I", "4":"H", "5":"V", "8":"E", "13":"E"}
cyclic_pep_re = dict([[v, k] for k, v in cyclic_pep.items()])
# cyclic_pep = {3:"I",}

def run(infile):

    with open(infile) as f:
        f1 = f.readlines()

    muts = []
    for i in f1:
        muts.append(i.replace("\n", "").split(","))       

    submit = open("submit.dat", "w") 

    for i in muts:
        resid_1 = i[0][:-3]
        mut_1 = i[0][-3:]
        resid_2 = i[1][:-3]
        mut_2 = i[1][-3:]
        file_name = cyclic_pep[resid_1] + resid_1 + three_one[mut_1] + cyclic_pep[resid_2] + resid_2 + three_one[mut_2]
        submit.write(file_name + "\n")
        if os.path.exists(os.path.join(".", file_name)):
            shutil.rmtree(os.path.join(".", file_name))
            os.makedirs(os.path.join(".", file_name))
        else:
            os.makedirs(os.path.join(".", file_name))

        os.chdir(os.path.join(".", file_name))
        cmd = "python " + os.path.join("..", "..", "Script", "mk_residue_mutate_with_psfgen_double_mut.py") + " " + i[0] + " " + i[1]
        os.system(cmd)
        time.sleep(1)
        cmd = "python " + os.path.join("..", "..", "Script", "mk_md_run_NAMD_double_mut.py") + " " + file_name
        os.system(cmd)
        time.sleep(1)
        shutil.move("complex.pdb", "./common/complex.pdb")
        shutil.move("complex.psf", "./common/complex.psf")
        os.chdir("..")
    submit.close()
            
def main():

    infile = sys.argv[1]
    run(infile)

if __name__=="__main__":
    main() 
