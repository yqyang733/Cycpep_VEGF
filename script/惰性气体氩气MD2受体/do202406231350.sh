mkdir Um
cd Um
cp -r ../top* .
cp ../index.ndx .
gmx distance -s ../pull/pull.tpr -f ../pull/pull.xtc -n index.ndx -select 'com of group "Backbone" plus com of group "lig"' -oall dist.xvg