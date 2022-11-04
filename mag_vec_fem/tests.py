import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from exploit_fun import *

pot_version=0
lnm=500
h=0.005
gauge='Sym'
l=lnm*10**-9
N_eig=100
res_path=res_path_f()
for pot_version in range(9):
    for lnm in (50,100,200,500,1000):
        nameV=res_path+'/pre_interp_pot/'+'pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'
        '''nameq=res_path+'/Vq/Vql'+str(lnm)+'.npy'
        Th=HyperCube(2,int(1/h),l=lnm*10**-9)
        V_preinterp=0.001*np.load(nameV)
        points=np.load(nameq)
        #V=interpolate_pot(V_preinterp,points,Th.q)
        print(np.shape(V_preinterp),np.shape(points))
        '''
        V_unscaled=np.load(nameV)
        V_scaled=0.001*V_unscaled
        np.save(nameV,V_scaled)