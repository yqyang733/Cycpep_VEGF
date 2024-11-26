# Cycpep_VEGF

# Temp
（1）使用package对单突，双突，三突，四突和五突的MD的轨迹进行特征提取。代码如下：  
```shell
for i in `cat triple_lst`;do mkdir ${i};cp /public/home/yqyang/VEGF/FEP/triple_mut_md/${i}/common/complex.psf ./${i}/;cp /public/home/yqyang/VEGF/FEP/triple_mut_md/${i}/prod/com-prodstep.dcd ./${i}/;done

```
