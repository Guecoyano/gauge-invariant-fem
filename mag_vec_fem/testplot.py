import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from exploit_fun import *

pot_version=2
lnm=500
h=0.005
gauge='Sym'
l=lnm*10**-9
N_eig=100
res_path=res_path_f()
nameV=res_path+'/pre_interp_pot/'+'pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'
nameq=res_path+'/Vq/Vql'+str(lnm)+'.npy'
Th=HyperCube(2,int(1/h),l=lnm*10**-9)
V_preinterp=np.load(nameV)
points=np.load(nameq)
V=interpolate_pot(V_preinterp,points,Th.q)

plt.figure()
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotIsolines(Th,V,N=40,fill=True,colorbar=True)
title='potential'+str(pot_version)
t1=title+'.png'
plt.title(title)
plt.savefig(res_path+t1)
plt.close()