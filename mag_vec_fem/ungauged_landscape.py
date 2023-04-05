# coding=utf-8
from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
import numpy as np
import pickle

res_pathldhsbc = data_path

d = 2
namepot = "Na400x15sig22v0"
h = 0.005
NB = 100
pot_version = 0
print("1. Set square mesh")
# Th=HyperCube(2,int(1/h),l=lnm*10**-9)
NV = 100
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
V1, Th = vth_data(h, namepot)
print("  -> Mesh sizes : nq=%d, nme=%d, nbe=%d" % (Th.nq, Th.nme, Th.nbe))
# for NV, NB in zip((100, 100, 100, 100), (20, 25, 30, 50)):
for NV, NB in zip((100,), (1,)):
    E_s = (NB**2) / 2
    V = (NV**2) * V1 + E_s
    print("2. 3. Set and solve BVP : 2D Magnetic Schrödinger")
    x = gi.getSol(Th=Th, B=0.0, V=V)

    print("4. Post-processing")

    # in the data name NB reps the shift through E_s=NB**2/2
    namedata = "h" + str(int(1 / h)) + namepot + "NV" + str(NV) + "NB" + str(NB)
    np.savez_compressed(
        os.path.realpath(os.path.join(res_pathldhsbc, "landscapes", namedata)),
        q=Th.q,
        u=x,
    )
