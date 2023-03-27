# coding=utf-8
from fem_base.exploit_fun import *
import pickle

res_path = data_path

N_eig = 10
N_a = 400
NB = 0
NV = 100
sigma = 2.2
h = 0.01
v = 0
gauge = "LandauX"
x = 0.15

B = NB**2
V_max = NV**2


with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
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

namepot = "v6"
V1, Th = vth_data(h, namepot, Th=Th)


get_eigplots(N_eig, B, namepot, V1 * V_max, V_max, Th, h, gauge, mass_lumping=True)
gauge = "Sym"
# get_eigplots(N_eig, B, namepot,V1 * V_max, V_max,  Th, h, gauge, mass_lumping=True)
