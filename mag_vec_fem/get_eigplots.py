from exploit_fun import *
res_path=res_path_f()

N_eig=20
lnm=200
B=30
V_maxmeV=5
h=0.005
pot_version=8
gauge='LandauX'
l=lnm*10**-9


VmeV,Th=vth_data(res_path,lnm,h,pot_version)


get_eigplots(N_eig,lnm,B,h,Th,VmeV*V_maxmeV,V_maxmeV,gauge,pot_version,res_path)