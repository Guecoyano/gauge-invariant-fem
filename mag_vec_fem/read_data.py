# coding=utf-8
from fem_base.exploit_fun import datafile, read_eigplot, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube
import os
import numpy as np

res_path = data_path
namepot = "Na400x15sig22v0"
V_max = 50
h = 0.001
pot_version = 0
gauge = "Sym"
N_eig = 100
n = 1
print("Building mesh")
Th = HyperCube(2, int(1 / h), l=1)

"""for b in (15,):
    for V_max in (1000,5000):
        for Num in (1,2,3,17,18,19,20,21,22,23,24,25,26,27,34,35,36,37,38,39,40,41,42,43,44,45):
            dat_file=datafile(res_path,lnm,b,V_max,h,pot_version,gauge,N_eig)
            name_preeig='l'+str(lnm)+'B'+str(b)+'V'+str(V_max)+'h'+str(int(1/h))+'v'+str(pot_version)+gauge
            save_eigplot(n,lnm,b,Th,dat_file,Num,'modulus',name_preeig,res_path)
            n+=1"""
plt.close("all")
for NV, NB in zip((100, 100, 100), (5, 10, 15)):
    for Num in (1, 2, 3, 4):
        print("Charging namepot, NV=" + str(NV) + " NB=" + str(NB))
        dat_file = datafile(namepot, NV, NB, gauge, h, N_eig)
        print("Reading")
        read_eigplot(n, namepot, NV, NB, Th, dat_file, Num, "modulus")
        n += 1
plt.show()
