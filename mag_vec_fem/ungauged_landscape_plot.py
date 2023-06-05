# coding=utf-8
from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
from fem_base.graphics import PlotVal, PlotBounds
import pickle
import numpy as np
import os

# parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("gauge", True, "Sym"),
    ("N_eig", False, 10),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("L", False, 200),
    ("beta", False, 0),
    ("eta", False, 0),
    ("plot_u", False, True),
    ("plot_w", False, True),
    ("namepot", True, None),
    ("load_file", True, None),
    ("dir_to_save", True, None),
    ("name_u", True, None),
    ("name_w", True, None),
]
h =L= (
    gauge
) = (
    N_eig
) = (
    N_a
) = (
    x
) = (
    sigma
) = (
    v
) = (
    beta
) = eta = namepot = load_file = dir_to_save = plot_u = plot_w = name_u = name_w = None
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
if name_u is None:
    name_u = f"u_{namepot}eta{eta}beta{beta}{gauge}h{int(1/h)}"
if name_w is None:
    name_w = f"w_{namepot}eta{eta}beta{beta}{gauge}h{int(1/h)}"
plt.close("all")


"""
target="0"
num=0
eta_str,beta_str="2e-1","2e-1"
name_u=f"u_{namepot}L{L}eta{eta_str}beta{beta_str}h{int(1/h)}"
subdiru="20230602-02"
diru=f"{data_path}/{subdiru}"
load_file=f"{diru}/{name_u}.npz"
DIR=f"{diru}"
dir_to_save=f"{DIR}/plots"
"""
print("1. Set square mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1/h)}.pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q=L*Th.q

# load_file=f"mag_vec_fem/data/merc5avril2023/u_h{int(1/h)}{namepot}NV{NV}NB{NB}.npz"

u = np.abs(np.load(load_file, allow_pickle=True)["u"])
# np.load(res_path + "/landscapes/h200" + namedata, allow_pickle=True)["u"]
print("size of u", np.shape(u))

print("5.   Plot")
epsilon = 10**-20


if plot_u:
    umin, umax = colorscale(u, "hypercube")
    save_name = os.path.realpath(os.path.join(dir_to_save, name_u))
    plt.figure()
    plt.clf()
    PlotBounds(Th, legend=False, color="k")
    plt.axis("off")
    PlotVal(Th, u, vmin=umin, vmax=umax)
    plt.title(f"Landscape function with shift $S={beta}$")
    plt.savefig(f"{save_name}.png")
    plt.clf()
    plt.close()

if plot_w:
    w = np.reciprocal(np.maximum(np.real(u), epsilon)) - beta
    wmin, wmax = colorscale(w, "hypercube")
    save_name = os.path.realpath(os.path.join(dir_to_save, name_w))
    plt.figure()
    plt.clf()
    PlotBounds(Th, legend=False, color="k")
    plt.axis("off")
    PlotVal(Th, w, vmin=wmin, vmax=wmax)
    plt.title(f"Effective potential minus the shift $S={beta}$")
    plt.savefig(f"{save_name}.png")
    plt.clf()
    plt.close()
