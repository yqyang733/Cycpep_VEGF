with open("lead_1.pdb") as f:
    f1 = f.readlines()

lst_C = []
lst_O = []
lst_N = []
lst_S = []
lst_Cl = []

for i in f1:
    if i[13:16].startswith("C"):
        lst_C.append(i)
    elif i[13:16].startswith("O"):
        lst_O.append(i)
    elif i[13:16].startswith("N"):
        lst_N.append(i)
    elif i[13:16].startswith("S"):
        lst_S.append(i)
    elif i[12:16].startswith("Cl"):
        lst_Cl.append(i)

alphabet_list = [chr(ord('A') + i) for i in range(26)]
lst_name = []
for i in alphabet_list:
    for j in alphabet_list:
        lst_name.append(i+j)

rt = open("lead_2.pdb", "w")

num = 0
for i in lst_C:
    rt.write(i[:13]+"C"+lst_name[num]+i[16:])
    num += 1

num = 0
for i in lst_O:
    rt.write(i[:13]+"O"+lst_name[num]+i[16:])
    num += 1

num = 0
for i in lst_N:
    rt.write(i[:13]+"N"+lst_name[num]+i[16:])
    num += 1

num = 0
for i in lst_S:
    rt.write(i[:13]+"S"+lst_name[num]+i[16:])
    num += 1

num = 0
for i in lst_Cl:
    rt.write(i[:12]+"Cl"+lst_name[num]+i[16:])
    num += 1