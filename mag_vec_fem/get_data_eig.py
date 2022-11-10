from exploit_fun import *
res_path=res_path_f()

lnm=200
h=0.001
gauge='Sym'
N_eig=100

B=1
V_maxmeV=10
for pot_version in (7,8):
    VmeV,Th=vth_data(res_path,lnm,h,pot_version)
    getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

B=5
for V_maxmeV in (1,10,100):
    for pot_version in (5,6,7,8):
        VmeV,Th=vth_data(res_path,lnm,h,pot_version)
        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

B=10
for pot_version in (7,8):
    for V_maxmeV in (1,10,100):
        VmeV,Th=vth_data(res_path,lnm,h,pot_version)
        for gauge in ('Sym','LandauX'):
            getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)