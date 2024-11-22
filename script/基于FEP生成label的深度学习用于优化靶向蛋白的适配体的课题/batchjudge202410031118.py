with open("test.txt") as f:
    f1 = f.readlines()
lst = []
for i in f1:
    lst.append(i.strip())
print(f"for i in {lst}:cmd.load(f'{{i}}/{{i}}.cif.gz')")


with open("test.txt") as f:
    f1 = f.readlines()
lst = ""
for i in f1:
    lst = lst + i.strip() + " "
print(f"for i in {lst};do python do1.py $i;done")