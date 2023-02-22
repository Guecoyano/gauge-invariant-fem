# coding=utf-8
from fem_base.exploit_fun import *
import os
from fem_base.exploit_fun import data_path

res_path = data_path


N_eig = 10
lnm = 100
B = 30
V_maxmeV = 0
h = 0.1
pot_version = 1
gauge = "Sym"
l = lnm * 10**-9


VmeV, Th = vth_data(lnm, h, pot_version)


getsave_eig(N_eig, lnm, B, VmeV, V_maxmeV, Th, h, pot_version, gauge)
