# coding=utf-8
from fem_base.exploit_fun import *
import os.path as op
from matplotlib import pyplot as plt
import pickle

res_path = data_path
lnm = 200
h = 0.001
N_a=400
x=0.15
sigma=2.2
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1/h)}.pkl")), "rb",
) as f:
    Th = pickle.load(f)
vq = np.load(os.path.join(res_path, "Vq", "VqNa400.npy"))
for v in (0,):
    N_a = 400
    namepot=f"Na{N_a}x{int(100*x)}sig{int(10*sigma)}v{v}"
    nameV = op.realpath(op.join(res_path, "pre_interp_pot", f"{namepot}.npy"))
    V_preinterp = np.load(nameV)

    # V_preinterp=1/np.max(V_preinterp)*V_preinterp
    # V1,Th=vth_data(lnm,h,pot_version,Th=Th)
    V = interpolate_pot(V_preinterp, vq, Th.q)
    plt.close()
    PlotIsolines(Th, 1/V, fill=True, colorbar=True)
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
