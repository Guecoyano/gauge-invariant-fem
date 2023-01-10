# coding=utf-8
from fem_base.exploit_fun import datafile, saveplots_fromdata, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube
import os
import numpy as np

res_path='/Volumes/Transcend/Thèse/mag_vec_fem/data' #data_path
N_a=400
x=0.05
sigma=2.2
gauge='Sym'
N_eig=100
h=0.001
Th=HyperCube(2,int(1/h),l=1)

B=1
V_max=10
for v in (7,8):
    namepot='Na'+str(N_a)+'x'+str(int(100*x))+'sig'+str(int(10*sigma))+'v'+str(v)
    namedata=os.path.realpath(os.path.join(res_path,'eigendata',namepot+'V'+str(V_max)+'B'+str(B)+gauge+'h'+str(int(1/h))+'Neig'+str(N_eig)+'.npz'))
    dat_file=np.load(namedata,allow_pickle=True)
    name_preeig=namepot+'V'+str(V_max)+'B'+str(B)+gauge+'h'+str(int(1/h))
    saveplots_fromdata(Th,dat_file,name_preeig)

"""gauge='Sym'
for pot_version in (0,):
    for V_max in (100,200,300,50):
        for B in (20,30,40):
            dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig,res_path=res_path)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
            print(name_preeig)
            saveplots_fromdata(Th,dat_file,name_preeig, res_path=data_path, phase=True)"""