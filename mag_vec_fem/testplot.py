# coding=utf-8
from fem_base.exploit_fun import datafile, save_eigplot, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube
import os
import numpy as np

# res_path='/Volumes/Transcend/Thèse/mag_vec_fem/data'
res_pathldhsbc = data_path
N_a = 400
x = 0.15
sigma = 2.2
gauge = "Sym"
N_eig = 100
h = 0.001
print("building mesh")
Th = HyperCube(2, int(1 / h), l=1)

NB = 1
NV = 10
B = NB**2
V_max = NV**2
for v in (0,):
    for NB in (1, 5, 10, 15, 20, 30):
        for NV in (200,):
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
            namedata = os.path.realpath(
                os.path.join(
                    res_pathldhsbc,
                    "eigendata",
                    namepot
                    + "NV"
                    + str(NV)
                    + "NB"
                    + str(NB)
                    + gauge
                    + "h"
                    + str(int(1 / h))
                    + "Neig"
                    + str(N_eig)
                    + ".npz",
                )
            )
            print("loading", namedata)
            dat_file = np.load(namedata, allow_pickle=True)
            name_preeig = (
                namepot
                + "NV"
                + str(NV)
                + "NB"
                + str(NB)
                + gauge
                + "h"
                + str(int(1 / h))
            )
            print("plotting")
            for n in range(11):
                save_eigplot(n, 200, Th, dat_file, n + 1, "modulus", name_preeig)
