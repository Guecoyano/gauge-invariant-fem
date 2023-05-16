# coding=utf-8
from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
from fem_base.graphics import PlotVal, PlotMesh, PlotBounds
import pickle
import numpy as np

res_path = data_path

plt.close("all")

h = 0.001
namepot = "Na400x15sig22v0"

print("1. Set square mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1/h)}.pkl")), "rb",
) as f:
    Th = pickle.load(f)
NV = 100
#V1, Th = vth_data(h, namepot)
#print("  -> Mesh sizes : nq=%d, nme=%d, nbe=%d" % (Th.nq, Th.nme, Th.nbe))
#u = []
# namedata = namepot + "NV" + str(NV) + "NB" + str(100) + ".npz"
# u += [np.load(res_path + "/landscapes/" + namedata, allow_pickle=True)["u"]]
for NB in (300,400):
    load_file=f"mag_vec_fem/data/merc5avril2023/u_h{int(1/h)}{namepot}NV{NV}NB{NB}.npz"
    """E_s=hbar*q_e*B/(2*m_e)
    V=V_max*V1+E_s
    print("2. 3. Set and solve BVP : 2D Magnetic Schrödinger")
    x=gi.getSol(Th=Th,B=0.0000000001,V=V)

    print("4. Post-processing")

    #in the data name B reps the shift through E_s=hbar*q_e*B/(2*m_e)
    namedata='l'+str(lnm)+'B'+str(B)+'V'+str(V_max)+'h'+str(int(1/h))+'v'+str(pot_version)
    np.savez_compressed(os.path.realpath(os.path.join(res_path,'landscapes',namedata)),q=Th.q,u=x)
    """

    namedata = namepot + "NV" + str(NV) + "NB" + str(NB) + ".npz"
    u = np.load(load_file, allow_pickle=True)["u"]
    #np.load(res_path + "/landscapes/h200" + namedata, allow_pickle=True)["u"]
    print("size of u", np.shape(u))

    print("5.   Plot")
    epsilon=10**-20
    E_0 = NB**2 / 2
    w=np.reciprocal(np.maximum(np.real(u), epsilon))-E_0
    wmin, wmax = colorscale(w, "hypercube")

    plt.figure(NB)
    plt.clf()
    PlotBounds(Th, legend=False, color="k")
    plt.axis("off")
    PlotVal(Th, w ,vmin=wmin,vmax=wmax)
    plt.title(r"Effective potential with shift $N_B=%d$ minus $E_0$" % (NB))
    t = "w_E0" + namepot + "NB" + str(NB) + "NV" + str(NV) + "h" + str(int(1 / h))
    # plt.savefig(os.path.realpath(os.path.join(res_path, t)))
    plt.show()
    plt.clf()
    plt.close()

    """plt.figure(NB + 1)
    plt.clf()
    PlotBounds(Th, legend=False, color="k")
    plt.axis("off")
    vmin, vmax = colorscale(u.real, "hypercube")
    PlotVal(Th, u.real, vmin=vmin, vmax=vmax)
    # matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plt.title(r"Shifted landscape $N_B=%d$" % (NB))
    t = "u_" + namepot + "NB" + str(NB) + "NV" + str(NV) + "h" + str(int(1 / h))
    # plt.savefig(os.path.realpath(os.path.join(res_path, t)))
    plt.show()
    plt.clf()
    plt.close()"""


""" 
plt.figure(NB+1)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,(u[2]-u[1]).real)
plt.title(r'u10-u5')
t='deltau_'+namepot+'NB'+str(NB)+'NV'+str(NV)+'h'+str(int(1/h))
#plt.clf()"""

"""plt.figure(3)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,np.abs(x))
plt.title(r'u0 l=$lnm=%d$'%(lnm))
plt.show()
plt.close()"""
