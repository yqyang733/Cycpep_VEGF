import sys
from molvs import standardize_smiles

def standard(file):
    with open(file) as f:
        f1 = f.readlines()
    output = ""
    for i in f1:
        name = i.split(",")[0]
        smiles = i.split(",")[1].strip()
        smiles_stand = standardize_smiles(smiles)
        output = output + name + "," + smiles_stand + "\n"
    with open("drug_smiles_standard.txt","w") as rt:
        rt.write(output)

def main():
    file = str(sys.argv[1])
    standard(file)

if __name__ == '__main__':
    main()