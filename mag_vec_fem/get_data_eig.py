from exploit_fun import *
res_path=res_path_f()

lnm=200
h=0.01
pot_version=4
gauge='Sym'
l=lnm*10**-9
N_eig=100
VmeV,Th=vth_data(res_path,lnm,h,pot_version)

for pot_version in (7,):
    VmeV,Th=vth_data(res_path,lnm,h,pot_version)
    for gauge in ('Sym', 'LandauX'):
        for B in (10,100):
            for V_maxmeV in (100,):
                getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

#lnm=200
#VmeV,Th=vth_data(res_path,lnm,h,pot_version)
#for B in (0,1,5,15,30,50):
#    N_eig=200
#    for V_maxmeV in (10,50,150):
#        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)

