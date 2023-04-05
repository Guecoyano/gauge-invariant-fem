# coding=utf-8
from fem_base.exploit_fun import *

res_pathldhsbc = data_path

lnm = 200
h = 0.01
gauge = "Sym"
N_eig = 10
print("Creating mesh")
Th = HyperCube(2, int(1 / h), l=1)
print("Mesh done")
B = 1
V_max = 10
N_a = 400
x = 0.15
sigma = 2.2
for v in (2,):
    for NB in (1, 100):
        for NV in (100,):
            namepot = (
                "Na"
                + str(N_a)
                + "x"
                + str(int(100 * x))
                + "sig"
                + str(int(10 * sigma))
                + "v"
                + str(v)
            )
            V1, Th = vth_data(h, namepot, Th=Th)
            getsave_eig(namepot, V1, NV, Th, h, NB, gauge, N_eig)

"""B=5
for V_max in (1,10,100):
    for pot_version in (5,6,7,8):
        namedata='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'h'+str(int(1/h))+'v'+str(pot_version)+gauge
        V1,Th=vth_data(lnm,h,pot_version,Th)
        getsave_eig(N_eig,lnm,B,V1,V_max,Th,h,pot_version,gauge)

B=10
for pot_version in (7,8):
    for V_max in (1,10,100):
        V1,Th=vth_data(lnm,h,pot_version,Th)
        for gauge in ('Sym','LandauX'):
            namedata='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'h'+str(int(1/h))+'v'+str(pot_version)+gauge
            getsave_eig(N_eig,lnm,B,V1,V_max,Th,h,pot_version,gauge)"""
