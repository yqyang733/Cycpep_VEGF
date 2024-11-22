import sys

def rename_chain_number(pdb):
    
    new_chain = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J", " K", " L", " M", " N", " O", " P", " Q", " R", " S", " T", " U", " V", " W", " X", " Y", " Z",]
    seg = ["PRA", "PRB", "PRC", "PRD", "PRE", "PRF", "PRG", "PRH", "PRI", "PRJ"]

    with open(pdb) as f:
        f1 = f.readlines()

    old_chains = list()
    for i in f1:
        if i.startswith("ATOM") or i.startswith("HETATM"):
            old_chains.append(i[20:22])
    # old_chains = list(old_chains)
    # old_chains.reverse()
    old_chains = sorted(set(old_chains), key=old_chains.index)
    print("old_chains: ", old_chains)

    cycles = len(old_chains)//len(new_chain)
    others = len(old_chains)%len(new_chain)
    new_seg_chains = []
    for i in range(cycles):
        for j in new_chain:
            new_seg_chains.append(seg[i]+j)
    for i in new_chain[:others]:
        new_seg_chains.append(seg[cycles]+i)

    print("new_seg_chains: ", new_seg_chains)

    dic_old_new = dict(zip(old_chains, new_seg_chains))

    rt = open(pdb.strip(".pdb")+"_rechain.pdb", "w")
    # idx = 1
    # for i in f1:
    #     if i.startswith("ATOM") or i.startswith("HETATM"):
    #         if idx <= 99999:
    #             rt.write(i[0:6]+'{:5d}'.format(idx)+i[11:20]+dic_old_new[i[20:22]][-2:]+i[22:72]+dic_old_new[i[20:22]][:3]+i[75:])
    #         else:
    #             rt.write(i[0:6]+'{:5d}'.format(99999)+i[11:20]+dic_old_new[i[20:22]][-2:]+i[22:72]+dic_old_new[i[20:22]][:3]+i[75:])
    #     else:
    #         rt.write(i)

    for i in f1:
        if i.startswith("ATOM") or i.startswith("HETATM"):
            rt.write(i[0:20]+dic_old_new[i[20:22]][-2:]+i[22:72]+dic_old_new[i[20:22]][:3]+i[75:])
        else:
            rt.write(i)

def main():

    rename_chain_number(sys.argv[1])

if __name__=="__main__":
    main() 