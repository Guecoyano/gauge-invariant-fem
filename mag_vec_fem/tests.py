# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
#from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
import os
#from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pickle


res_path = data_path
path=os.path.realpath(os.path.join(res_path,"film_poles"))
h = 0.01
gauge = "Sym"
N_eig = 1
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")
N_a = 200
x = 0.15
sigma = 2.2
v = 0
nframes=15
NBmax,NBmin=10,0
'''nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))
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
V1, Th = vth_data(h, namepot, Th=Th,N_a=200)'''
