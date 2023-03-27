# coding=utf-8
from fem_base.exploit_fun import *
import os.path as op
from matplotlib import pyplot as plt

res_path = data_path
lnm = 200
h = 0.005
Th = HyperCube(2, int(1 / h), l=1)
vq = np.load(os.path.join(res_path, "Vq", "VqNa400.npy"))
for pot_version in (0,):
    N_a = 400
    """nameV = op.realpath(
        op.join(
            res_path,
            "pre_interp_pot",
            "Na400x15sig22v" + str(pot_version) + ".npy",
        )
    )"""
    nameV = op.realpath(op.join(res_path, "pre_interp_pot", "v6.npy"))
    V_preinterp = np.load(nameV)

    # V_preinterp=1/np.max(V_preinterp)*V_preinterp
    # V1,Th=vth_data(lnm,h,pot_version,Th=Th)
    V = interpolate_pot(V_preinterp, vq, Th.q)
    plt.close()
    PlotIsolines(Th, V, fill=True, colorbar=True)
    # plt.imshow(V,cmap='turbo')
    plt.show()
    plt.clf()
    plt.close()

    """plt.figure(2)
    plt.clf()
    PlotBounds(Th,legend=False,color='k')
    plt.axis('off')
    PlotVal(Th,V1)
    plt.title(r'Potential version $v=%d$ for $lnm=%d$'%(pot_version,lnm))
    plt.show()
    #plt.savefig(os.path.realpath(os.path.join(res_path,'potentials','v'+str(pot_version)+'lnm'+str(lnm)+'.png')))
    plt.clf()
    plt.close()"""
