# coding=utf-8
from fem_base.exploit_fun import datafile, saveplots_fromdata, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube

res_path=data_path
lnm=200
h=0.001
gauge='Sym'
N_eig=100
Th=HyperCube(2,int(1/h),l=lnm*10**-9)

B=1
V_maxmeV=10
for pot_version in (7,8):
    dat_file=datafile(lnm,B,V_maxmeV,h,pot_version,gauge,N_eig)
    name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_maxmeV)+'v'+str(pot_version)+'h'+str(int(1/h))
    saveplots_fromdata(Th,dat_file,name_preeig)

B=5
for V_maxmeV in (1,10,100):
    for pot_version in (5,6,7,8):
        dat_file=datafile(lnm,B,V_maxmeV,h,pot_version,gauge,N_eig)
        name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_maxmeV)+'v'+str(pot_version)+'h'+str(int(1/h))
        saveplots_fromdata(Th,dat_file,name_preeig)
B=10
for pot_version in (7,8):
    for V_maxmeV in (1,10,100):
        for gauge in ('Sym','LandauX'):
            dat_file=datafile(lnm,B,V_maxmeV,h,pot_version,gauge,N_eig)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_maxmeV)+'v'+str(pot_version)+'h'+str(int(1/h))
            saveplots_fromdata(Th,dat_file,name_preeig)