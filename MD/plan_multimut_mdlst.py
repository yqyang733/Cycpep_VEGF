import sys

one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}
three_letter = dict([[v,k] for k,v in one_letter.items()])
three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
'D':'ASP', 'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 'Y':'TYR',    \
'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',    \
'G':'GLY', 'P':'PRO', 'C':'CYS'}

dict_site_3 = {"3MET":-1.15,"3LEU":-0.62}
dict_site_4 = {"4CYS":-1.43, "4PHE":-0.63, "4ILE":-0.47, "4VAL":-0.94, "4TYR":-0.55}
dict_site_5 = {"5ALA":-0.82, "5CYS":-0.46, "5LYS":-0.55, "5SER":-0.9, "5THR":-0.99}
dict_site_8 = {"8TRP":-2.72, "8VAL":-1.77, "8THR":-1.47, "8ILE":-0.92, "8ASN":-0.77}
dict_site_13 = {"13LEU":-2.57, "13VAL":-2.02, "13GLN":-2.02, "13TRP":-1.68, "13ILE":-1.66}

dict_site_all = {**dict_site_3, **dict_site_4, **dict_site_5, **dict_site_8, **dict_site_13}

def generate_number(string):
    number = ""
    res = ""
    for char in string:
        if char.isdigit():
            number += char
        else:
            res += char
    return int(number), res

def run(in_f, rank):

    with open(in_f) as f:
        f1 = f.readlines()
    
    all_plan = []
    
    for i in f1:
        site_mut = dict()
        name_ = i.strip().split("_")
        for a in name_:
            resi, resname = generate_number(a)
            site_mut[resi] = (resname, a)
        dict_new_tmp = dict()
        for b in dict_site_all.keys():
            i_resi, i_resname = generate_number(b)
            if i_resi in site_mut.keys():
                pass
            else:
                dict_new_tmp[(i_resi, i_resname)] = dict_site_all[b]
        sorted_dict = dict(sorted(dict_new_tmp.items(), key=lambda item: item[1]))
        
        for resi, resname in list(sorted_dict.keys())[:int(rank)]:
            new_dict_sitemut = dict(site_mut)
            new_dict_sitemut[resi] = (resname, str(resi)+resname)
            new_dict_sitemut.update()
            sorted_dict_mut = dict(sorted(new_dict_sitemut.items(), key=lambda item: item[0]))
            print(sorted_dict_mut)
            all_mut = []
            for aa in sorted_dict_mut:
                all_mut.append(sorted_dict_mut[aa][1])
            all_plan.append("_".join(all_mut))

    all_plan = list(set(all_plan))
    rt = open("mdplanlst.dat", "w")
    rt.write("\n".join(all_plan))
    rt.close()

def main():

    in_f = sys.argv[1]
    rank = sys.argv[2]
    run(in_f, rank)
    
if __name__=="__main__":
    main() 