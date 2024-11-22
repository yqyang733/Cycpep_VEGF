import os

site_3 = ["3MET", "3LEU"]
site_4 = ["4CYS", "4PHE", "4ILE", "4VAL", "4TYR"]
site_5 = ["5ALA", "5CYS", "5LYS", "5SER", "5THR"]
site_8 = ["8TRP", "8VAL", "8THR", "8ILE", "8ASN"]
site_13 = ["13LEU", "13VAL", "13GLN", "13TRP", "13ILE"]

def run():

    submit_3_5813 = open("mut_3_5813.dat", "w")
    for i in site_3:
        for j in site_5+site_8+site_13:
            submit_3_5813.write(i+","+j+"\n")
    submit_3_5813.close()

    submit_4_813 = open("mut_4_813.dat", "w")
    for i in site_4:
        for j in site_8+site_13:
            submit_4_813.write(i+","+j+"\n")
    submit_4_813.close()

    submit_5_13 = open("mut_5_13.dat", "w")
    for i in site_5:
        for j in site_13:
            submit_5_13.write(i+","+j+"\n")
    submit_5_13.close()

def main():
    run()

if __name__=="__main__":
    main()
