line=`wc -l ../ZINC_all.smi| awk '{print int($1/100)+1}'`
split -l ${line} ../ZINC_all.smi -d -a 2 ZINC_