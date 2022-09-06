import numpy as np
from fem_base.gaugeInvariantFEM import gauss_pot
from fem_base import mesh


for l in [50,100,200,500]:
    for h in [0.1]:
        T=mesh.HyperCube(2,int(1/h),l=l*10**-9)
        name='Documents//Thèse loc EHQ/AaTh'+str(int(1/h))+'l'+str(l)
        np.save(name,T)