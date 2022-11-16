# coding=utf-8
from fem_base.exploit_fun import *
import os.path as op
from matplotlib import pyplot as plt
res_path=data_path
lnm=200
pot_version=5
nameV=op.realpath(op.join(res_path,'pre_interp_pot','pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'))
V_preinterp=np.load(nameV)

plt.close()
plt.imshow(V_preinterp, interpolation='bilinear')
plt.show()
