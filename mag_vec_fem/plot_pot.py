# coding=utf-8
from fem_base.exploit_fun import *
import os.path as op
from matplotlib import pyplot as plt
res_path=data_path
lnm=200
h=0.001
Th=HyperCube(2,int(1/h),l=1)
for pot_version in range(9):
    #nameV=op.realpath(op.join(res_path,'pre_interp_pot','pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'))
    #V_preinterp=np.load(nameV)
    V1,Th=vth_data(lnm,h,pot_version,Th=Th)
    '''plt.close()
    plt.imshow (V1)
    plt.show()
    plt.clf()
    plt.close()'''

    plt.figure(2)
    plt.clf()
    PlotBounds(Th,legend=False,color='k')
    plt.axis('off')
    PlotVal(Th,V1)
    plt.title(r'Potential version $v=%d$ for $lnm=%d$'%(pot_version,lnm))
    #plt.show()
    plt.savefig(os.path.realpath(os.path.join(res_path,'potentials','v'+str(pot_version)+'lnm'+str(lnm)+'.png')))
    plt.clf()
    plt.close()