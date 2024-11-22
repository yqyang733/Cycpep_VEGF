
namd_fep_result(){
    file_in=$1
    cd ${file_in}
    echo "bonded,ddg" >> bonded.dat
    cd bonded
    for i in `seq 1 3`
    do
        cd dup${i}/prod
        ddg=`grep "#Free energy" complex-prod.fepout| tail -1| awk -F "now is " '{print $2}'`
        echo "dup_${i},${ddg}" >> ../../../bonded.dat
        cd ../../
    done
    cd ..

    echo "free,ddg" >> free.dat
    cd free
    for i in `seq 1 3`
    do
        cd dup${i}/prod
        ddg=`grep "#Free energy" complex-prod.fepout| tail -1| awk -F "now is " '{print $2}'`
        echo "dup_${i},${ddg}" >> ../../../free.dat
        cd ../../
    done
    cd ..

    cat > do_ddg.py << EOF
def calculate_ddg(bonded, free):
    import numpy as np
    import math

    complex_fep = []
    with open(bonded) as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        complex_fep.append(f1[i].strip().split(",")[1])
    RNA_fep = []
    with open(free) as f:
        f1 = f.readlines()
    for i in range(1, len(f1)):
        RNA_fep.append(f1[i].strip().split(",")[1])

    complex_fep = np.array(complex_fep, dtype=np.float64)
    RNA_fep = np.array(RNA_fep, dtype=np.float64)
    complex_mean = np.mean(complex_fep)
    RNA_mean = np.mean(RNA_fep)
    complex_se = np.std(complex_fep, ddof=1)/math.sqrt(len(complex_fep))
    RNA_se = np.std(RNA_fep, ddof=1)/math.sqrt(len(RNA_fep))
    ddg_mean = complex_mean - RNA_mean
    ddg_se = math.sqrt(complex_se*complex_se + RNA_se*RNA_se)
    print("The mean of ddg (kcal/mol):", ddg_mean)
    print("The SE of ddg (kcal/mol):", ddg_se)
    # print("The mean of ddg (kcal/mol):", ddg_mean/4.184)
    # print("The SE of ddg (kcal/mol):", ddg_se/4.184)
    result = open("ddg_result.txt", "w")
    # result.write("The ddg (kJ/mol): " + str(ddg_mean) + " +/- " + str(ddg_se) + "\n")
    result.write("The ddg (kcal/mol): " + str(ddg_mean) + " +/- " + str(ddg_se) + "\n")

def main():
    import sys

    calculate_ddg(sys.argv[1], sys.argv[2])

if __name__=="__main__":
    main()
EOF

    python do_ddg.py bonded.dat free.dat
    cd ..
}

input=$*
namd_fep_result ${input}