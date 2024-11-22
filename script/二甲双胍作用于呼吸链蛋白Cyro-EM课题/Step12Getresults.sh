for i in `cat wins.dat`;do echo ${i};tail -2 ${i}/prod/prodf_all.xvg;done

mkdir analysis
cd analysis
for i in `seq 0 25`;do a=`echo ${i}*4|bc`;echo win_${a} >> wins.dat;done
for i in `cat wins.dat`;do cp ../../${i}/prod/prod.tpr ./pmf${i}.tpr;done
for i in `cat wins.dat`;do echo pmf${i}.tpr >> tpr-files.dat;done
for i in `cat wins.dat`;do cp ../../${i}/prod/prodf_all.xvg ./pmf${i}_pullf_all.xvg;done
for i in `cat wins.dat`;do echo pmf${i}_pullf_all.xvg >> prodf-files.dat;done
gmx wham -it tpr-files.dat -if prodf-files.dat -o -unit kCal -bsres zerror.xvg -bsprof zerrorprofile.xvg -nBootstrap 20 -bs-method b-hist -temp 303 -b 5000 -e 10000

for i in `seq 0 19`;do cp ./win_${i}/prod/prod.tpr ./analysis/pmf${i}.tpr;done
for i in `seq 0 19`;do echo pmf${i}.tpr >> tpr-files.dat;done
for i in `seq 0 19`;do cp ./win_${i}/prod/prodf.xvg ./analysis/pmf${i}_pullf.xvg;done
for i in `seq 0 19`;do echo pmf${i}_pullf.xvg >> prodf-files.dat;done
gmx wham -it tpr-files.dat -if prodf-files.dat -o -unit kCal -bsres zerror.xvg -bsprof zerrorprofile.xvg -nBootstrap 20 -bs-method b-hist -temp 303

mkdir analysis
cd analysis
mkdir run1
cd run1
cat ../../submit_1.dat ../../submit_2.dat ../../submit_3.dat > wins.dat
for i in `cat wins.dat`;do cp ../../${i}/prod/prod.tpr ./pmf${i}.tpr;done
for i in `cat wins.dat`;do echo pmf${i}.tpr >> tpr-files.dat;done
for i in `cat wins.dat`;do cp ../../${i}/prod/prodf.xvg ./pmf${i}_pullf.xvg;done
for i in `cat wins.dat`;do echo pmf${i}_pullf.xvg >> prodf-files.dat;done
gmx wham -it tpr-files.dat -if prodf-files.dat -o -unit kCal -bsres zerror.xvg -bsprof zerrorprofile.xvg -nBootstrap 20 -bs-method b-hist -temp 303 -b 5000

cp ../../Umbrella_3/analysis/run1/wins.dat .
mv wins.dat wins_forward.dat
for i in `cat wins_forward.dat`;do cp ../../Umbrella_3/${i}/prod/prod.tpr ./pmf${i}_forward.tpr;done
for i in `cat wins_forward.dat`;do echo pmf${i}_forward.tpr >> tpr-files.dat;done
for i in `cat wins_forward.dat`;do cp ../../Umbrella_3/${i}/prod/prodf.xvg ./pmf${i}_pullf_forward.xvg;done
for i in `cat wins_forward.dat`;do echo pmf${i}_pullf_forward.xvg >> prodf-files.dat;done
cp ../../Umbrella_4_1/analysis/run1/wins.dat .
mv wins.dat wins_backward.dat
for i in `cat wins_backward.dat`;do cp ../../Umbrella_4_1/${i}/prod/prod.tpr ./pmf${i}_backward.tpr;done
for i in `cat wins_backward.dat`;do echo pmf${i}_backward.tpr >> tpr-files.dat;done
for i in `cat wins_backward.dat`;do cp ../../Umbrella_4_1/${i}/prod/prodf.xvg ./pmf${i}_pullf_backward.xvg;done
for i in `cat wins_backward.dat`;do echo pmf${i}_pullf_backward.xvg >> prodf-files.dat;done
gmx wham -it tpr-files.dat -if prodf-files.dat -o -unit kCal -bsres zerror.xvg -bsprof zerrorprofile.xvg -nBootstrap 20 -bs-method b-hist -temp 303 -b 2000