# coding=utf-8
from fem_base.exploit_fun import variable_value, saveplots_fromdata, data_path
import os
import numpy as np
import pickle
import sys

#parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("gauge", True, "Sym"),
    ("N_eig", False, 10),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("NB", False, 0),
    ("NV", False, 0),
    ("namepot", True, None),
    ("load_file", True, None),
    ("dir_to_save", True, None),
    ("name_eig", True, None),
]
h=gauge=N_eig=N_a=x=sigma=v=NB=NV=namepot=load_file=dir_to_save=name_eig=None
print(sys.argv)

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default
if namepot is None:
    namepot=(f"Na{N_a}x{int(100*x)}sig{int(10*sigma)}v{v}")
if dir_to_save is None:
    dir_to_save = os.path.join(data_path, "eigenplots")
if name_eig is None:
    name_eig=(f"{namepot}NV{NV}NB{NB}{gauge}h{int(1/h)}Neig{N_eig}"
    )
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1 / h)}.pkl")),
    "rb",
) as f:
    Th = pickle.load(f)

dat_file = np.load(load_file, allow_pickle=True)
saveplots_fromdata(Th, dat_file, name_eig, dir_to_save=dir_to_save, phase=True)