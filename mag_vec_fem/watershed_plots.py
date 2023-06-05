"""script to use watershed from a landscape computed on a mesh

the landscape is interpolated on a grid via P1 elements, and the grid should be chosen so as to be finer than the mesh (2x finer for instance)"""

import numpy as np
import os
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm
import pickle
from fem_base.graphics import PlotIsolines


#parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("L", False, 200),
    ("N_a", False, 400),
    ("N_eig", False, 10),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("beta", False, 5e-2),
    ("eta", False, 2e-1),
    ("gauge", True, "Sym"),
    ("namepot", True, None),
    ("file_u", True, None),
    ("file_eig", True, None),
    ("dir_to_save", True, None),
    ("name_ws", True, None),
    ("E_barr", False, None),
    ("coeff", False, 0),
    ("which_cond", True, "energy_limit"),#"standard_cond", "shifted_cond"
]
h=L=gauge=N_eig=N_a=x=sigma=v=beta=eta=namepot=file_u=file_eig=None
plot_diff=plot_w=name_u=dir_to_save=name_w=which_cond=E_barr=coeff=None
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
    name_u=(f"u_{namepot}eta{eta}beta{beta}h{int(1/h)}"
    )
if name_w is None:
    name_w=(f"w_{namepot}eta{eta}beta{beta}h{int(1/h)}"
    )
if E_barr is None:
    E_barr=beta+coeff*eta


target="0"
num=0
eta_str,beta_str="1e-3","0"
eta=eval(eta_str)
beta=eval(beta_str)
E_barr=0#eval(beta_str)-eval(eta_str)
coeff=1.
#name_base=f"{namepot}eta{eta_str}beta{beta_str}"
name_eig=f"{namepot}eta{eta_str}beta{beta_str}{gauge}target{target}h{int(1/h)}Neig{N_eig}"
name_u=f"u_{namepot}L{L}eta{eta_str}beta{beta_str}h{int(1/h)}"
subdiru="20230601"
subdireig="20230601"
diru=f"{data_path}/{subdiru}"
direig=f"{data_path}/{subdireig}"
file_u=f"{diru}/{name_u}.npz"
file_eig=f"{direig}/{name_eig}.npz"
DIR=f"{direig}"
dir_to_save=f"{DIR}/plots"

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
u = np.real(epsilon+np.load(file_u, allow_pickle=True)["u"])
psi = np.log10(epsilon+np.abs(np.load(file_eig, allow_pickle=True)["eig_vec"][:,num]))

if len(u) != Th.nq:
    print("u does not have th.nq values")
    exit()

w=1/u

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

wmin_cs,wmax_cs=colorscale(w,"hypercube")

print("Plotting boundaries over effective potential")
x = []
y = []
z = []
for n in range(Th.nq):
    if len(merged.global_boundary[n]) >= 2:
        x.append(Th.q[n, 0])
        y.append(Th.q[n, 1])
        z.append(min(w[n],wmax_cs)-wmin_cs)
        
z=np.array(z)
zmin,zmax=0,0.002
print(wmax_cs-wmin_cs)
plt.figure()
plt.clf()
wmin, wmax = -20,1
#PlotVal(Th, 1 / u, vmin=vmin, vmax=vmax)

#PlotIsolines(Th, psi, fill=True, colorbar=True, vmin=wmin, vmax=wmax, color="turbo")
plt.tricontourf(
    Th.q[:, 0],
    Th.q[:, 1],
    Th.me,
    psi,
    10,
    cmap="turbo",
    vmin=wmin,
    vmax=wmax,
    )
plt.colorbar()
plt.gca().set_aspect("equal")
plt.axis("off")
plt.scatter(x, y, c='k',alpha=np.sqrt((z/zmax)),vmin=zmin,vmax=zmax, cmap="Greys", s=(200*(z))**2)
plt.show()
plt.clf()
plt.close()