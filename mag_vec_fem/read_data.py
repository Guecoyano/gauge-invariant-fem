# coding=utf-8
from fem_base.exploit_fun import datafile, read_eigplot, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube

res_path=data_path
lnm=200
V_maxmeV=50
h=0.01
pot_version=7
gauge='Sym'
N_eig=100
n=1
Th=HyperCube(2,int(1/h),l=lnm*10**-9)

'''for b in (15,):
    for V_maxmeV in (1000,5000):
        for Num in (1,2,3,17,18,19,20,21,22,23,24,25,26,27,34,35,36,37,38,39,40,41,42,43,44,45):
            dat_file=datafile(res_path,lnm,b,V_maxmeV,h,pot_version,gauge,N_eig)
            name_preeig='l'+str(lnm)+'B'+str(b)+'V'+str(V_maxmeV)+'h'+str(int(1/h))+'v'+str(pot_version)+gauge
            save_eigplot(n,lnm,b,Th,dat_file,Num,'modulus',name_preeig,res_path)
            n+=1'''
plt.close('all')
for b in (10,):
    for Num in (1,2,3,4,50,53,54,55,56,70,71,72,73,74,75,80):
        for V_maxmeV in (100,):
            dat_file=datafile(lnm,b,V_maxmeV,h,pot_version,gauge,N_eig)
            read_eigplot(n,lnm,b,Th,dat_file,Num,'modulus')
            n+=1
plt.show()