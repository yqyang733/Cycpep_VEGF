get_fep_rt(){
    cd bonded_in
    for i in `seq 1 3`;do cd dup${i};mkdir temp;cd temp;for a in `seq 0 29`;do cp ../lambda${a}/prod/dhdl.xvg dhdl_lambda${a}.xvg;done;gmx bar -f dhdl_lambda*.xvg -oi -o > fep.log;cd ../..;done
    for i in `seq 1 3`;do tail -2 ./dup${i}/temp/fep.log|head -1|awk '{print $6}' >> ../bonded.dat;done
    cd ../free_in
    for i in `seq 1 3`;do cd dup${i};mkdir temp;cd temp;for a in `seq 0 29`;do cp ../lambda${a}/prod/dhdl.xvg dhdl_lambda${a}.xvg;done;gmx bar -f dhdl_lambda*.xvg -oi -o > fep.log;cd ../..;done
    for i in `seq 1 3`;do tail -2 ./dup${i}/temp/fep.log|head -1|awk '{print $6}' >> ../free.dat;done
    cd ..
}

cal_ddg(){

    cat > ddg.py << EOF
import math
import numpy as np

def calculate_ddg():
    bonded = []
    with open("bonded.dat") as f:
        f1 = f.readlines()
    for i in range(len(f1)):
        bonded.append(f1[i].strip())
    free = []
    with open("free.dat") as f:
        f1 = f.readlines()
    for i in range(len(f1)):
        free.append(f1[i].strip())

    bonded = np.array(bonded, dtype=np.float64)
    free = np.array(free, dtype=np.float64)
    bonded_mean = np.mean(bonded)
    free_mean = np.mean(free)
    bonded_se = np.std(bonded, ddof=1)/math.sqrt(len(bonded))
    free_se = np.std(free, ddof=1)/math.sqrt(len(free))
    ddg_mean = bonded_mean - free_mean
    ddg_se = math.sqrt(bonded_se*bonded_se + free_se*free_se)
    print("The mean of ddg (kJ/mol):", ddg_mean)
    print("The SE of ddg (kJ/mol):", ddg_se)
    print("The mean of ddg (kcal/mol):", ddg_mean/4.184)
    print("The SE of ddg (kcal/mol):", ddg_se/4.184)
    result = open("ddg_result.txt", "w")
    result.write("The ddg (kcal/mol): " + str(ddg_mean/4.184) + " +/- " + str(ddg_se/4.184) + "\n")

calculate_ddg()
EOF

    python ddg.py
}

get_fep_rt
cal_ddg
