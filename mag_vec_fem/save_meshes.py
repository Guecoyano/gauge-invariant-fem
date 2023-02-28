# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.exploit_fun import *
import os
import numpy as np
import pickle


for h in (0.01, 0.001, 0.005):
    Th = HyperCube(2, int(1 / h), l=1)
    # np.save(data_path+'testh',Th,allow_pickle=True)
    with open(
        os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
        "wb",
    ) as f:
        pickle.dump(Th, f)
"""
h=0.001
with open(os.path.realpath(os.path.join(data_path,'Th','h'+str(int(1/h))+'.pkl')),'rb') as f:
    x = pickle.load(f)
print(x.nme)
"""
