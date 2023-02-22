# coding=utf-8
from fem_base.exploit_fun import *

res_path = data_path

N_eig = 20
lnm = 200
B = 30
V_max = 5
h = 0.005
pot_version = 8
gauge = "LandauX"


V1, Th = vth_data(lnm, h, pot_version)


get_eigplots(N_eig, lnm, B, h, Th, V1 * V_max, V_max, gauge, pot_version)
