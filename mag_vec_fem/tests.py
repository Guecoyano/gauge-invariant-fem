import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from exploit_fun import *
from os import *

pot_version=0
lnm=200
h=0.005
gauge='Sym'
l=lnm*10**-9
N_eig=100
res_path=res_path_f()
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

h=0.055
B=1
for pot_version in (7,):
    for V_maxmeV in (1,):
        for gauge in ('LandauX',):
            VmeV,Th=vth_data(res_path,lnm,h,pot_version)
            getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

            dat_file=datafile(res_path,lnm,B,V_maxmeV,h,pot_version,gauge,N_eig)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_maxmeV)+'v'+str(pot_version)+'h'+str(int(1/h))
            saveplots_fromdata(Th,dat_file,name_preeig,res_path)