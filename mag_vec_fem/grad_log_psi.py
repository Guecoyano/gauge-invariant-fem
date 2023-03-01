"""
This script aims at computing the grad(log(modulus psi)) to check wether there is an exponential decay when crossing regions of a landscape. 
"""
# coding=utf-8
import numpy as np
from fem_base.exploit_fun import *
import os
import matplotlib
import matplotlib.pyplot as plt
from fem_base.graphics import *
from scipy.interpolate import griddata
from watershed_landscape import *

# matplotlib.scale.SymmetricalLogScale(axis, *, base=10, linthresh=2, subs=None, linscale=1)
# ax.set_yticks([0, 50, 100, 150])

res_path = data_path

def modgradlogpsi(psi, points, eps=0, gpoints=200):
    grid_x, grid_y = np.mgrid[-0.5 : 0.5 : gpoints * 1j, -0.5 : 0.5 : gpoints * 1j]
    psigrid = np.array(griddata(points, psi, (grid_x, grid_y)))
    n = gpoints
    logpsi = np.log(np.abs(psigrid) + eps)
    gx = np.zeros((n, n))
    gy = np.zeros((n, n))
    for i in range(n - 2):
        gx[i + 1, :] = (logpsi[i + 2, :] - logpsi[i, :]) / (2 * n)
        gy[:, i + 1] = (logpsi[:, i + 2] - logpsi[:, i]) / (2 * n)
    gx[0, :] = (logpsi[1, :] - logpsi[0, :]) / (n)
    gx[n - 1, :] = (logpsi[n - 1, :] - logpsi[n - 2, :]) / (n)
    gy[:, 0] = (logpsi[:, 1] - logpsi[:, 0]) / (n)
    gy[:, n - 1] = (logpsi[:, n - 1] - logpsi[:, n - 2]) / (n)

    return np.sqrt(gx**2 + gy**2)


def logmodpsi(psi, points, eps=0, gpoints=200):
    grid_x, grid_y = np.mgrid[-0.5 : 0.5 : gpoints * 1j, -0.5 : 0.5 : gpoints * 1j]
    psigrid = np.array(griddata(points, np.log(np.abs(psi)+eps), (grid_x, grid_y)))
    n = gpoints
    return psigrid


N_a, x, sig, v, gauge, h, N_eig = 400, 15, 22, 0, "Sym", 0.001, 100
NV, NB = 100, 30
num = 1
namepot = "Na" + str(N_a) + "x" + str(x) + "sig" + str(sig) + "v" + str(v)

"""#gradlog of the potential
namedata=os.path.realpath(os.path.join(res_path,'pre_interp_pot',namepot+'.npy'))
dat_file=np.ravel(np.load(namedata,allow_pickle=True))
points=np.load(os.path.join(res_path,'Vq','VqNa400.npy'))
#name_preeig=namepot+'NV'+str(NV)+'NB'+str(NB)+gauge+'h'+str(int(1/h))
gradlogv=modgradlogpsi(dat_file,points,gpoints=400)
print(np.shape(gradlogv))

f=plt.figure()
plt.clf()
im=plt.imshow(gradlogv.T,origin='lower',cmap='gist_heat')
cbar = plt.colorbar(im)
cbar.set_label("$grad(log(modulus(\psi )))$ ")
plt.show()
plt.clf()
plt.close()"""
print("creating mesh")
Th = HyperCube(2, int(1 / h), l=1)
#####################################
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)

print("load u")
nameu = os.path.realpath(
    os.path.join(
        data_path,
        "landscapes",
        "h200"+"Na400x15sig22v0NV" + str(NV) + "NB" + str(NB) + ".npz",
    )
)
u = np.load(nameu, allow_pickle=True)["u"]
if len(u) != Th.nq:
    print("u does not have th.nq values")
    exit()
# we make sure we can invert u by adding epsilon
epsilon = 10**-20
u = np.real(u + epsilon)

print("vertex structuration")
vertex_w = ws.vertices_data_from_mesh(Th, values=(1 / u))

print("find local minima")
M = ws.label_local_minima(vertex_w, mode="irr")

print("initial watershed")
W = ws.watershed(vertex_w, M, mode="irr")

print("initial numberof regions:", np.max(M))

print("Structuring data")
regions = wsm.init_regions(vertex_w, M, W)


print("merging regions")


def cond(min, barr):
    return barr > coeff * min


def shifted_cond(min, barr):
    return barr - E > coeff * (min - E)


merged = wsm.merge_algorithm(regions, shifted_cond)
x = []
y = []
for n in range(Th.nq):
    if len(regions.global_boundary[n]) >= 2:
        x.append(Th.q[n, 0])
        y.append(Th.q[n, 1])
############################################
print("mesh done")
for NB,num in zip((30,30,30),(1,7,5)):
    namedata = os.path.realpath(
    os.path.join(
        res_path,
        "eigendata",
        namepot
        + "NV"
        + str(NV)
        + "NB"
        + str(NB)
        + gauge
        + "h"
        + str(int(1 / h))
        + "Neig"
        + str(N_eig)
        + ".npz",
    )
    )
    psi = np.load(namedata)["eig_vec"][:, num-1]
    
    gradlog = np.minimum(modgradlogpsi(psi, Th.q, gpoints=400, eps=10**-10), 0.0001*(4+2*NB/30))
    logpsi = logmodpsi(psi, Th.q, gpoints=400, eps=10**-5)
    plt.clf()
    plt.close()
    f1 = plt.subplot(1, 2, 1)
    plt.scatter(x, y, c="k", s=2)
    plt.imshow(gradlog.T, origin="lower", cmap="gist_heat")  # ,cmap='gist_heat'
    # cbar1 = plt.colorbar(im1)
    # cbar1.set_label("$grad(log(modulus(\psi )))$ ")
    f2 = plt.subplot(1, 2, 2)
    plt.scatter(x, y, c="k", s=2)
    im2 = plt.imshow(logpsi.T, origin="lower", cmap="turbo")  # ,cmap='gist_heat'
    # cbar2 = plt.colorbar(im2)
    # cbar2.set_label("$log(modulus(\psi ))$ ")
    t="gradlogpsi_NV"+str(NV)+"NB"+str(NB) +"boundaries"
    plt.savefig(os.path.realpath(os.path.join(res_path, t)))
    plt.clf()
    plt.close()
