# coding=utf-8
from fem_base.exploit_fun import datafile, saveplots_fromdata, data_path
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube
import os
import numpy as np
import pickle

# res_path='/Volumes/Transcend/Thèse/mag_vec_fem/data'
res_path = data_path
N_a = 400
x = 0.15
sigma = 2.2
gauge = "Sym"
N_eig = 10
h = 0.01
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)

NB = 1
NV = 10
B = NB**2
V_max = NV**2
for v in (0,):
    for NB in (10, 0):
        frame = int(NB / 10)
        for NV in (100,):
            """namepot = (
                "Na"
                + str(N_a)
                + "x"
                + str(int(100 * x))
                + "sig"
                + str(int(10 * sigma))
                + "v"
                + str(v)
            )"""
            namepot = "v6"
            """namedata = os.path.realpath(
                os.path.join(
                    res_path,
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
            )"""
            namedata = os.path.realpath(
                os.path.join(
                    res_path,
                    "film_poles",
                    namepot
                    + "NV"
                    + str(NV)
                    + "NBmin0NBmax10"
                    + gauge
                    + "h"
                    + str(int(1 / h))
                    + "Neig"
                    + str(N_eig)
                    + "frame"
                    + str(frame)
                    + ".npz",
                )
            )
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
            saveplots_fromdata(Th, dat_file, name_preeig, phase=True)

"""gauge='Sym'
for pot_version in (0,):
    for V_max in (100,200,300,50):
        for B in (20,30,40):
            dat_file=datafile(lnm,B,V_max,h,pot_version,gauge,N_eig,res_path=res_path)
            name_preeig='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'v'+str(pot_version)+'h'+str(int(1/h))+gauge
            print(name_preeig)
            saveplots_fromdata(Th,dat_file,name_preeig, res_path=data_path, phase=True)"""
