# coding=utf-8
from fem_base.exploit_fun import *
res_path=data_path

lnm=200
h=0.001
gauge='Sym'
N_eig=100

B=1
V_max=10
for pot_version in (7,8):
    V1,Th=vth_data(lnm,h,pot_version)
    getsave_eig(N_eig,lnm,B,V1,V_max,Th,h,pot_version,gauge)

B=5
for V_max in (1,10,100):
    for pot_version in (5,6,7,8):
        V1,Th=vth_data(lnm,h,pot_version)
        getsave_eig(N_eig,lnm,B,V1,V_max,Th,h,pot_version,gauge)

B=10
for pot_version in (7,8):
    for V_max in (1,10,100):
        V1,Th=vth_data(lnm,h,pot_version)
        for gauge in ('Sym','LandauX'):
            getsave_eig(N_eig,lnm,B,V1,V_max,Th,h,pot_version,gauge)