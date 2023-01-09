# coding=utf-8
from fem_base.exploit_fun import datafile, saveplots_fromdata, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube

res_path='/Volumes/Transcend/Thèse/mag_vec_fem/data' #data_path
lnm=200
h=0.001
gauge='Sym'
N_eig=100
Th=HyperCube(2,int(1/h),l=1)

'''B=1
V_max=10
for pot_version in (7,8):
    dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig)
    name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
    saveplots_fromdata(Th,dat_file,name_preeig)

B=5
for V_max in (1,10,100):
    for pot_version in (5,6,7,8):
        dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig)
        name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
        saveplots_fromdata(Th,dat_file,name_preeig)
B=10
for pot_version in (7,8):
    for V_max in (1,10,100):
        for gauge in ('Sym','LandauX'):
            dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig,res_path=res_path)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
            saveplots_fromdata(Th,dat_file,name_preeig, res_path=res_path, real=True)
'''
gauge='Sym'
for pot_version in (0,):
    for V_max in (100,200,300,50):
        for B in (20,30,40):
            dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig,res_path=res_path)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
            print(name_preeig)
            saveplots_fromdata(Th,dat_file,name_preeig, res_path=data_path, phase=True)