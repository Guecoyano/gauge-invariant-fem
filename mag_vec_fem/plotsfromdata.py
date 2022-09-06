from exploit_fun import datafile, saveplots_fromdata, res_path_f
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube

res_path=res_path_f()
lnm=100
V_maxmeV=50
h=0.002
pot_version=1
gauge='Sym'
N_eig=100
n=0
Th=HyperCube(2,int(1/h),l=lnm*10**-9)
for b in (30,):
    for V in (150,):
        dat_file=datafile(res_path,lnm,b,V_maxmeV,h,pot_version,gauge,N_eig)
        saveplots_fromdata(Th,dat_file,'l100B30V150v1h500',res_path)
        n+=1

