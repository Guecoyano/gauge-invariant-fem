# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
from os import *

pot_version=0
lnm=200
h=0.05
gauge='Sym'
l=lnm*10**-9
N_eig=100
res_path=data_path
#E_0=hbar*q_e*B/(2*m_e)
'''for pot_version in range(9):
    for lnm in (50,100,200,500,1000):
        nameV=res_path+'/pre_interp_pot/'+'pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'
        nameq=res_path+'/Vq/Vql'+str(lnm)+'.npy'
        Th=HyperCube(2,int(1/h),l=lnm*10**-9)
        V_preinterp=0.001*np.load(nameV)
        points=np.load(nameq)
        #V=interpolate_pot(V_preinterp,points,Th.q)
        print(np.shape(V_preinterp),np.shape(points))
        
        V_unscaled=np.load(nameV)
        V_scaled=0.001*V_unscaled
        np.save(nameV,V_scaled)
absol_path=path.abspath("")
print(getcwd())
print(dirname(''))
print(res_path)'''

h=0.045
B=10
E_0=hbar*q_e*B/(2*m_e)
t=0
for pot_version in (6,):
    for V_maxmeV in (10,10,10,10,10,10,10,10,10,10,10,10):
        for gauge in ('LandauX',):
            VmeV,Th=vth_data(lnm,h,pot_version)
            #getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,target_energy=6*10**-22)
            t+=1

            dat_file=datafile(lnm,B,V_maxmeV,h,pot_version,gauge,N_eig)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_maxmeV)+'v'+str(pot_version)+'h'+str(int(1/h))
            saveplots_fromdata(Th,dat_file,name_preeig+"t"+str(t),phase=True)