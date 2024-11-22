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

def generate_bool_list(string):
    bool_list = []
    for char in string:
        if char.isdigit():
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list

with open("doublefep.csv") as f:
    f1 = f.readlines()
rt = open("triplemdplan.csv","w")
for i in f1:
    name_ = i.strip()
    judge = generate_bool_list(name_)
    for a in range(len(judge)):
        if a >= 1:
            if judge[a]==True and judge[a-1]==False and judge[a+1]==False:
                flag = a+2
                break
            elif judge[a]==True and judge[a-1]==True and judge[a+1]==False:
                flag = a+2
                break
    mut_1 = name_[:flag]
    mut_2 = name_[flag:]
    mut_1_site = mut_1[1:-1]
    mut_2_site = mut_2[1:-1]
    dict_new_tmp = dict()
    for i in dict_site_all.keys():
        if i.startswith(mut_1_site) or i.startswith(mut_2_site):
            pass
        else:
            dict_new_tmp[i] = dict_site_all[i]
    sorted_dict = dict(sorted(dict_new_tmp.items(), key=lambda item: item[1]))
    for i in range(5):
        rt.write(mut_1_site+three_letter[mut_1[-1]]+","+mut_2_site+three_letter[mut_2[-1]]+","+list(sorted_dict.keys())[i]+"\n")
