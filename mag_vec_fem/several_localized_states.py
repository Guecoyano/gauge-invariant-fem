# coding=utf-8
from fem_base.exploit_fun import *
import os
import numpy as np
import pickle
import sys

# parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("L", False, 200),
    ("gauge", True, "Sym"),
    ("N_eig", False, 10),
    ("eig_list", False, [1, 2, 3, 4, 5]),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("beta", False, 0),
    ("eta", False, 0),
    ("namepot", True, None),
    ("load_file", True, None),
    ("dir_to_save", True, None),
    ("name_eig", True, None),
]
h = L = gauge = N_eig = N_a = x = sigma = v = beta = eta = namepot =None
load_file = dir_to_save = name_eig = eig_list = None
print(sys.argv)

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default
if namepot is None:
    namepot = f"Na{N_a}x{int(100*x)}sig{int(10*sigma)}v{v}"
if dir_to_save is None:
    dir_to_save = os.path.join(data_path, "eigenplots")
if name_eig is None:
    name_eig = f"{namepot}eta{eta}beta{beta}{gauge}h{int(1/h)}Neig{N_eig}"
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1 / h)}.pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
Th.q = L * Th.q

dat_file = np.load(load_file, allow_pickle=True)
x = dat_file["eig_vec"]
eigs = np.abs(x[:, eig_list])
mod_tot = np.sum(eigs, 1)
plt.figure()
plt.axis("off")
PlotIsolines(Th, mod_tot, fill=True, colorbar=True, color="turbo")
plt.show()
plt.close()
plt.clf()
