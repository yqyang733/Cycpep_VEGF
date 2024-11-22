echo 4|gmx genrestr -f solv_ions.gro -o posre_1.itp
echo -e "; Include Position restraint file\n#ifdef POSRES_1\n#include \"posre_1.itp\"\n#endif\n" >> Protein_chain_A.itp

cp posre_LIG.itp posre_LIG_1.itp
echo -e "; Include Position restraint file\n#ifdef POSRES_1\n#include \"posre_LIG_1.itp\"\n#endif\n" >> LIG.itp

cd mdp/
cp prod.mdp prod_1.mdp
# prod_1.mdp文件中加上位置限制。

cp job.sh job_1.sh
sbatch job_1.sh
# 修改提交脚本文件并提交任务。