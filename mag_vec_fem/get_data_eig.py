# coding=utf-8
from fem_base.exploit_fun import *
res_path=data_path

lnm=200
h=0.110
gauge='Sym'
N_eig=100

B=1
V_maxmeV=10
for pot_version in (7,8):
    VmeV,Th=vth_data(lnm,h,pot_version)
    getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge)

B=5
for V_maxmeV in (1,10,100):
    for pot_version in (5,6,7,8):
        VmeV,Th=vth_data(lnm,h,pot_version)
        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge)

B=10
for pot_version in (7,8):
    for V_maxmeV in (1,10,100):
        VmeV,Th=vth_data(lnm,h,pot_version)
        for gauge in ('Sym','LandauX'):
            getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge)