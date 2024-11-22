def similarity_zinc_topol(zinc, out):
    from rdkit import Chem
    from rdkit import DataStructs

    rt = open(out, "w")
    log = open(out+".log", "w")
    i = 0
    ref_smi = Chem.MolFromSmiles("CC(/C=C/N(C)C)=O")
    fp_ref = Chem.RDKFingerprint(ref_smi)
    with open("../ZINC/ZINC_100/"+zinc) as f:
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

def main():
    import sys

    zinc = sys.argv[1]
    out = sys.argv[2]
    similarity_zinc_topol(zinc, out)

if __name__=="__main__":
    main() 