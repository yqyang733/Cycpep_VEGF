import os
import sys
import shutil

class config:

    def __init__(self):

        self.bonded = os.path.join("/public/home/yqyang/file/fep_zhangd/mdp-Protein_1")   
        self.free = os.path.join("/public/home/yqyang/file/fep_zhangd/mdp-Protein_1")  
        self.dups = int(3)  

def do_mdp(b_f, dups):

    with open(os.path.join(b_f, "lambdas.dat")) as f:
        f_lambda = f.readlines()
    with open(os.path.join(b_f, "em.mdp")) as f:
        f_em = f.readlines()
    with open(os.path.join(b_f, "nvt.mdp")) as f:
        f_nvt = f.readlines()
    with open(os.path.join(b_f, "npt.mdp")) as f:
        f_npt = f.readlines()
    with open(os.path.join(b_f, "md.mdp")) as f:
        f_md = f.readlines()

    windows_num = len(f_lambda[0].strip("\n;").split())
    for i in range(dups):
        if os.path.exists(os.path.join(".", "dup"+str(i+1))):
            shutil.rmtree(os.path.join(".", "dup"+str(i+1)))
            os.makedirs(os.path.join(".", "dup"+str(i+1)))
        else:
            os.makedirs(os.path.join(".", "dup"+str(i+1)))
        for j in range(windows_num):
            os.makedirs(os.path.join(".", "dup"+str(i+1), "lambda"+str(j)))
            em_rt = open(os.path.join(".", "dup"+str(i+1), "lambda"+str(j), "em.mdp"), "w")
            for ii in f_em:
                if ii.startswith("init-lambda-state"):
                    em_rt.write(ii.replace("\n", "")+" "+str(j) + "\n" + f_lambda[0])
                elif ii.startswith("vdw-lambdas"):
                    em_rt.write(f_lambda[1])
                elif ii.startswith("coul-lambdas"):
                    em_rt.write(f_lambda[2])
                elif ii.startswith("bonded-lambdas"):
                    em_rt.write(f_lambda[3])
                elif ii.startswith("mass-lambdas"):
                    em_rt.write(f_lambda[4])
                elif ii.startswith("fep-lambdas"):
                    em_rt.write(f_lambda[5])
                else:
                    em_rt.write(ii)
            em_rt.close()

            nvt_rt = open(os.path.join(".", "dup"+str(i+1), "lambda"+str(j), "nvt.mdp"), "w")
            for ii in f_nvt:
                if ii.startswith("init-lambda-state"):
                    nvt_rt.write(ii.replace("\n", "")+" "+str(j) + "\n" + f_lambda[0])
                elif ii.startswith("vdw-lambdas"):
                    nvt_rt.write(f_lambda[1])
                elif ii.startswith("coul-lambdas"):
                    nvt_rt.write(f_lambda[2])
                elif ii.startswith("bonded-lambdas"):
                    nvt_rt.write(f_lambda[3])
                elif ii.startswith("mass-lambdas"):
                    nvt_rt.write(f_lambda[4])
                elif ii.startswith("fep-lambdas"):
                    nvt_rt.write(f_lambda[5])
                else:
                    nvt_rt.write(ii)
            nvt_rt.close()

            npt_rt = open(os.path.join(".", "dup"+str(i+1), "lambda"+str(j), "npt.mdp"), "w")
            for ii in f_npt:
                if ii.startswith("init-lambda-state"):
                    npt_rt.write(ii.replace("\n", "")+" "+str(j) + "\n" + f_lambda[0])
                elif ii.startswith("vdw-lambdas"):
                    npt_rt.write(f_lambda[1])
                elif ii.startswith("coul-lambdas"):
                    npt_rt.write(f_lambda[2])
                elif ii.startswith("bonded-lambdas"):
                    npt_rt.write(f_lambda[3])
                elif ii.startswith("mass-lambdas"):
                    npt_rt.write(f_lambda[4])
                elif ii.startswith("fep-lambdas"):
                    npt_rt.write(f_lambda[5])
                else:
                    npt_rt.write(ii)
            npt_rt.close()

            md_rt = open(os.path.join(".", "dup"+str(i+1), "lambda"+str(j), "md.mdp"), "w")
            for ii in f_md:
                if ii.startswith("init-lambda-state"):
                    md_rt.write(ii.replace("\n", "")+" "+str(j) + "\n" + f_lambda[0])
                elif ii.startswith("vdw-lambdas"):
                    md_rt.write(f_lambda[1])
                elif ii.startswith("coul-lambdas"):
                    md_rt.write(f_lambda[2])
                elif ii.startswith("bonded-lambdas"):
                    md_rt.write(f_lambda[3])
                elif ii.startswith("mass-lambdas"):
                    md_rt.write(f_lambda[4])
                elif ii.startswith("fep-lambdas"):
                    md_rt.write(f_lambda[5])
                else:
                    md_rt.write(ii)
            md_rt.close()
            
def run(state):

    settings = config()
    if state == "b":
        do_mdp(settings.bonded, settings.dups)
    elif state == "f":
        do_mdp(settings.free, settings.dups)                                                                                                                                                         

def main():

    state = sys.argv[1]
    run(state)
    
if __name__=="__main__":
    main() 
