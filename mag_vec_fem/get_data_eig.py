from exploit_fun import *
res_path=res_path_f()

lnm=100
h=0.002
pot_version=1
gauge='Sym'
l=lnm*10**-9


VmeV,Th=vth_data(res_path,lnm,h,pot_version)
for B in (5,15):
    N_eig=100
    for V_maxmeV in (1000,5000):
        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

#lnm=200
#VmeV,Th=vth_data(res_path,lnm,h,pot_version)
#for B in (0,1,5,15,30,50):
#    N_eig=200
#    for V_maxmeV in (10,50,150):
#        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

