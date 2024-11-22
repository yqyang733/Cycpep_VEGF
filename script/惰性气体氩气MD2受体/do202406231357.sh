mkdir analysis
cd analysis
mkdir run1
cd run1
cp ../../submit.dat .
for i in `cat submit.dat`;do cp ../../${i}/prod/prod.tpr ./pmf${i}.tpr;done
for i in `cat submit.dat`;do echo pmf${i}.tpr >> tpr-files.dat;done
for i in `cat submit.dat`;do cp ../../${i}/prod/prodf.xvg ./pmf${i}_pullf.xvg;done
for i in `cat submit.dat`;do echo pmf${i}_pullf.xvg >> prodf-files.dat;done
gmx wham -it tpr-files.dat -if prodf-files.dat -o -unit kCal -bsres zerror.xvg -bsprof zerrorprofile.xvg -nBootstrap 20 -bs-method b-hist -temp 310 -b 40000 -e 50000    