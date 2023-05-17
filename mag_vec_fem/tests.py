# coding=utf-8
import fem_base.gaugeInvariantFEM as gi

# from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
import os

# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pickle


res_pathldhsbc = data_path
path = os.path.realpath(os.path.join(res_pathldhsbc, "film_poles"))
h = 0.001
gauge = "Sym"
N_eig = 1
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")
N_a = 400
x = 0.15
sigma = 2.2
v = 0
"""nframes = 15
NBmax, NBmin = 10, 0
nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))"""
namepot = f"Na{N_a}x{int(100 * x)}sig{int(10 * sigma)}v{v}mean1"
V1, Th = vth_data(h, namepot, Th=Th)
print(np.mean(V1))
