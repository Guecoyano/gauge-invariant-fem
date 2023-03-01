# coding=utf-8
from fem_base.exploit_fun import *
import pickle

res_path = data_path
h = 0.001
gauge = "Sym"
N_eig = 5
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")
N_a = 400
x = 0.15
sigma = 2.2
v = 0
nframes=51
NBmax,NBmin=0,15
nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))

namepot = (
            "Na"
            + str(N_a)
            + "x"
            + str(int(100 * x))
            + "sig"
            + str(int(10 * sigma))
            + "v"
            + str(v)
        )
V1, Th = vth_data(h, namepot, Th=Th)
for NB in nbs:
    for NV in (100,):
        path=os.path.realpath(os.path.join(res_path,"film_poles"))
        getsave_eig(namepot, V1, NV, Th, h, NB, gauge, N_eig, dir_to_save=path)
        E_s = (NB**2) / 2
        V = (NV**2) * V1 + E_s
        x = gi.getSol(Th=Th, B=0.0, V=V)
        namedata = "u_h" + str(int(1 / h)) + namepot + "NV" + str(NV) + "frame"+ str(int(np.round(NB**2*(nframes-1)/(NBmax-NBmin-1),0)))
        np.savez_compressed(
            os.path.join(path, namedata),
            q=Th.q,
            u=x,
        )