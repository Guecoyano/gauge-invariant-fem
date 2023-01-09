# coding=utf-8
from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
from fem_base.graphics import PlotVal,PlotMesh,PlotBounds,FillMesh
from matplotlib.text import Text
from math import cos
import numpy as np

res_path=data_path

plt.close("all")

d=2
lnm=200
h=0.001
B=10
pot_version=0
print("1. Set square mesh")
#Th=HyperCube(2,int(1/h),l=lnm*10**-9)
V_max=100
V1,Th=vth_data(lnm,h,pot_version)
print("  -> Mesh sizes : nq=%d, nme=%d, nbe=%d" % (Th.nq,Th.nme,Th.nbe));
for B in (30,40):
    E_s=B/2
    V=V_max*V1+E_s
    print("2. 3. Set and solve BVP : 2D Magnetic Schrödinger")
    x=gi.getSol(Th=Th,B=0.0,V=V)
    
    print("4. Post-processing")

    #in the data name B reps the shift through E_s=hbar*q_e*B/(2*m_e)
    namedata='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'h'+str(int(1/h))+'v'+str(pot_version)
    np.savez_compressed(os.path.realpath(os.path.join(res_path,'landscapes',namedata)),q=Th.q,u=x)

    '''print('5.   Plot')
    plt.figure(1)
    plt.clf()
    PlotBounds(Th,legend=False,color='k')
    plt.axis('off')
    PlotVal(Th,x.real)
    plt.title(r'2D Magnetic Schrödinger : solution (real part) (mesh $n_q=%d$, $n_{me}=%d$)'%(Th.nq,Th.nme))


    plt.figure(2)
    plt.clf()
    PlotBounds(Th,legend=False,color='k')
    plt.axis('off')
    PlotVal(Th,x.imag)
    plt.title(r'2D Magnetic Schrödinger : solution (imaginary part) (mesh $n_q=%d$, $n_{me}=%d$)'%(Th.nq,Th.nme))


plt.figure(3)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,np.abs(x))
plt.title(r'u0 l=$lnm=%d$'%(lnm))
plt.show()
plt.close()'''