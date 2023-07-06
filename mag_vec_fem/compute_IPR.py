from IPR_computation import *
from fem_base.exploit_fun import load_mesh, variable_value
import sys

params = [
    ("h", False, 0.001),
    ("L", False, 200),
    ("load_file", True, None),
]
h=L=load_file=None

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default

Th=load_mesh(h,L)
num=0
eig_vec=np.load(load_file, allow_pickle=True)["eig_vec"][:,num]
ipr=IPR(Th,eig_vec)