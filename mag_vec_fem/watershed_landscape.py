"""script to use watershed from a landscape computed on a mesh

the landscape is interpolated on a grid via P1 elements, and the grid should be chosen so as to be finer than the mesh (2x finer for instance)"""

import numpy as np
import os
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm
import pickle
from fem_base.graphics import PlotVal, PlotIsolines


#parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("L", False, 200),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("beta", False, 0),
    ("eta", False, 0),
    ("plot_u",False,True),
    ("plot_w",False,True),
    ("namepot", True, None),
    ("load_file", True, None),
    ("dir_to_save", True, None),
    #("name_u", True, None),
    ("name_w", True, None),
    ("E_barr", False, None),
    ("coeff", False, 0),
    ("which_cond", True, "energy_limit"),#"standard_cond", "shifted_cond"
]
h=L=gauge=N_eig=N_a=x=sigma=v=beta=eta=namepot=load_file=None
plot_u=plot_w=name_u=dir_to_save=name_w=which_cond=E_barr=coeff=None
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
if name_u is None:
    name_u=(f"u_{namepot}eta{eta}beta{beta}{gauge}h{int(1/h)}"
    )
if name_w is None:
    name_w=(f"w_{namepot}eta{eta}beta{beta}{gauge}h{int(1/h)}"
    )
if E_barr is None:
    E_barr=beta+coeff*eta



print("load mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q=L*Th.q
    Th.vols=(L**2)*Th.vols

print("load u")
epsilon = 10**-20
u = np.real(epsilon+np.load(load_file, allow_pickle=True)["u"])
if len(u) != Th.nq:
    print("u does not have th.nq values")
    exit()

w=1/u
wmin,wmax=colorscale(w,"hypercube")

print("vertex structuration")
vertex_w = ws.vertices_data_from_mesh(Th, values=w)

print("find local minima")
M = ws.label_local_minima(vertex_w, mode="irr")

print("initial watershed")
W = ws.watershed(vertex_w, M, mode="irr")

print("initial numberof regions:", np.max(M))

print("Structuring data")
regions = wsm.init_regions(vertex_w, M, W)


print("merging regions")

def energy_limit(min,barr):
    return (barr>E_barr and barr-beta>coeff*(min-beta))

def standard_cond(min, barr):
    return barr > coeff * min

def shifted_cond(min, barr):
    return barr - beta > coeff * (min - beta)

condition=globals()[which_cond]
merged = wsm.merge_algorithm(regions, condition)

final_min = 0
for r in merged.regions:
    if not r.removed:
        final_min += 1

print("final number of regions:", final_min)

print("Plotting boundaries over effective potential")
x = []
y = []
z = []
for n in range(Th.nq):
    if len(regions.global_boundary[n]) >= 2:
        x.append(Th.q[n, 0])
        y.append(Th.q[n, 1])
        z.append(min(w[n],wmax))
zmin,zmax=np.min(np.array(z)),np.max(np.array(z))
print(zmin,zmax)
plt.figure()
plt.clf()
wmin, wmax = colorscale(w, "hypercube")
#PlotVal(Th, 1 / u, vmin=vmin, vmax=vmax)

diff_w=np.abs(E_barr-w)
diff_min,diff_max=colorscale(diff_w,"hypercube")

PlotIsolines(Th, w, fill=True, colorbar=True, vmin=wmin, vmax=wmax, color="turbo")

plt.scatter(x, y, c=z, cmap="Greys", s=(50*(z-zmin))**2)
plt.show()
plt.clf()
plt.close()