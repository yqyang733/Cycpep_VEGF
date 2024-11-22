from molvs import standardize_smiles

lead = standardize_smiles("CNC(CN1C[C@H](c2c(C1=O)ccc(Cl)c2)C(Nc(cnc3)c4c3cccc4)=O)=O")

with open("opt2.csv") as f:
    f1 = f.readlines()

rt = open("input_pairs_1.csv", "w")
for i in f1:
    rt.write(lead+","+i)
rt.close()