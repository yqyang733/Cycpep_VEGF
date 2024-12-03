

dict_site_all = {**dict_site_3, **dict_site_4, **dict_site_5, **dict_site_8, **dict_site_13}

def generate_bool_list(string):
    bool_list = []
    for char in string:
        if char.isdigit():
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list

with open("4mut.csv") as f:
    f1 = f.readlines()
rt = open("5mdplan.csv","w")
all_plan = []
for i in f1:
    name_ = i.strip()
    # judge = generate_bool_list(name_)
    # num_tmp = []
    # for a in range(len(judge)):
    #     if a >= 1 and a <= len(judge)-2:
    #         print(judge[a], judge[a+1])
    #         if judge[a]==False and judge[a+1]==True:
    #             num_tmp.append(a)
    # print(num_tmp)
    # mut_1 = name_[:num_tmp[0]+1]
    # mut_2 = name_[num_tmp[0]+1:num_tmp[1]+1]
    # mut_3 = name_[num_tmp[1]+1:]
    # print(mut_1, mut_2, mut_3)
    # if mut_1[1].isdigit():
    #     mut_1_site = int(mut_1[0:2])
    # else:
    #     mut_1_site = int(mut_1[0])
    # if mut_2[1].isdigit():
    #     mut_2_site = int(mut_2[0:2])
    # else:
    #     mut_2_site = int(mut_2[0])
    # if mut_3[1].isdigit():
    #     mut_3_site = int(mut_3[0:2])
    # else:
    #     mut_3_site = int(mut_3[0])

    name_lst = name_.split("_")
    if len(name_lst[0]) == 4:
        mut_1_site = int(name_lst[0][0])
        mut_1 = name_lst[0]
    elif len(name_lst[0]) == 5:
        mut_1_site = int(name_lst[0][0:2])
        mut_1 = name_lst[0]
    if len(name_lst[1]) == 4:
        mut_2_site = int(name_lst[1][0])
        mut_2 = name_lst[1]
    elif len(name_lst[1]) == 5:
        mut_2_site = int(name_lst[1][0:2])
        mut_2 = name_lst[1]
    if len(name_lst[2]) == 4:
        mut_3_site = int(name_lst[2][0])
        mut_3 = name_lst[2]
    elif len(name_lst[2]) == 5:
        mut_3_site = int(name_lst[2][0:2])
        mut_3 = name_lst[2]
    if len(name_lst[3]) == 4:
        mut_4_site = int(name_lst[3][0])
        mut_4 = name_lst[3]
    elif len(name_lst[3]) == 5:
        mut_4_site = int(name_lst[3][0:2])
        mut_4 = name_lst[3]

    dict_new_tmp = dict()
    for i in dict_site_all.keys():
        if i.startswith(str(mut_1_site)) or i.startswith(str(mut_2_site)) or i.startswith(str(mut_3_site)) or i.startswith(str(mut_4_site)):  # 这个有问题需要修改。
            pass
        else:
            dict_new_tmp[i] = dict_site_all[i]
    sorted_dict = dict(sorted(dict_new_tmp.items(), key=lambda item: item[1]))
    
    for i in range(len(list(sorted_dict.keys()))):
        
        judge = generate_bool_list(list(sorted_dict.keys())[i])
        num_tmp = []
        for a in range(len(judge)):
            if a >= 0 and a <= len(judge)-2:
                print(judge[a], judge[a+1])
                if judge[a]==True and judge[a+1]==False:
                    num_tmp.append(a)
        print(num_tmp)
        dict_mut = dict({mut_1_site:mut_1,mut_2_site:mut_2,mut_3_site:mut_3,mut_4_site:mut_4,int(list(sorted_dict.keys())[i][:num_tmp[0]+1]):list(sorted_dict.keys())[i],})
        print(type(dict_mut))
        print(dict_mut)
        print(dict_mut.items())
        sorted_dict_mut = dict(sorted(dict_mut.items(), key=lambda item: item[0]))
        print(sorted_dict_mut)
        # print(str(list(sorted_dict_mut.keys())[0])+sorted_dict_mut[list(sorted_dict_mut.keys())[0]][-3:])
        all_plan.append(str(list(sorted_dict_mut.keys())[0])+sorted_dict_mut[list(sorted_dict_mut.keys())[0]][-3:]+"_"+str(list(sorted_dict_mut.keys())[1])+sorted_dict_mut[list(sorted_dict_mut.keys())[1]][-3:]+"_"+str(list(sorted_dict_mut.keys())[2])+sorted_dict_mut[list(sorted_dict_mut.keys())[2]][-3:]+"_"+str(list(sorted_dict_mut.keys())[3])+sorted_dict_mut[list(sorted_dict_mut.keys())[3]][-3:]+"_"+str(list(sorted_dict_mut.keys())[4])+sorted_dict_mut[list(sorted_dict_mut.keys())[4]][-3:])
all_plan = list(set(all_plan))

rt.write("\n".join(all_plan))

def run():

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


def main():
    run()
    
if __name__=="__main__":
    main() 