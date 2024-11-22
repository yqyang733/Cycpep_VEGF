bond_free_build(){
    mut_before=${1}
    resi=${2}
    mut_after=${3}
    file=${mut_before}${resi}${mut_after}
    mkdir ${file}
    cd ${file}
    mkdir bonded_in free_in
    cd bonded_in
    cp ../../bonded_in/bonded.pdb .
    pmx mutate -f bonded.pdb -o mutant.pdb --keep_resid << -EOF
8
A
${resi}
${mut_after}
n
-EOF
    gmx pdb2gmx -f mutant.pdb -o conf.pdb -p topol.top -water tip3p << -EOF
7
-EOF
    pmx gentop -p topol.top -o newtop.top
    gmx editconf -f conf.pdb -o box.gro -bt cubic -d 0.5 
    gmx solvate -cp box -cs spc216 -p newtop -o water.gro
    gmx grompp -f /public/home/yqyang/file/gmx_file/ions.mdp -c water.gro -p newtop.top -o genion.tpr
    gmx genion -s genion.tpr -p newtop.top -neutral -conc 0.15 -o ions.gro << -EOF
SOL
-EOF
    gmx make_ndx -f ions.gro -o index.ndx << -EOF
q
-EOF

    cp ../../mk_FEP_lambda_mdp.py .
    python mk_FEP_lambda_mdp.py b
    for i in `seq 1 3`;do cp ../../job.parrallel ./dup${i};done

    box_size=`tail -1 box.gro|awk '{print $1}'`
    cd ../free_in
    cp ../../free_in/free.pdb .
    pmx mutate -f free.pdb -o mutant.pdb --keep_resid << -EOF
8
A
${resi}
${mut_after}
n
-EOF
    gmx pdb2gmx -f mutant.pdb -o conf.pdb -p topol.top -water tip3p << -EOF
7
-EOF
    pmx gentop -p topol.top -o newtop.top
    gmx editconf -f conf.pdb -o box.gro -bt cubic -box ${box_size} ${box_size} ${box_size}
    gmx solvate -cp box -cs spc216 -p newtop -o water.gro
    gmx grompp -f /public/home/yqyang/file/gmx_file/ions.mdp -c water.gro -p newtop.top -o genion.tpr
    gmx genion -s genion.tpr -p newtop.top -neutral -conc 0.15 -o ions.gro << -EOF
SOL
-EOF
    gmx make_ndx -f ions.gro -o index.ndx << -EOF
q
-EOF
  
    cp ../../mk_FEP_lambda_mdp.py .
    python mk_FEP_lambda_mdp.py b
    for i in `seq 1 3`;do cp ../../job.parrallel ./dup${i};done

    cd ..
}

input=$*
bond_free_build ${input}   # sh .sh T 4 D
