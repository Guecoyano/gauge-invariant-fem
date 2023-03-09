"""
This script aims at computing the grad(log(modulus psi)) to check wether there is an exponential decay when crossing regions of a landscape. 
"""
# coding=utf-8
import numpy as np
from fem_base.exploit_fun import *
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from fem_base.graphics import *
from scipy.interpolate import griddata
import pickle
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm
#from watershed_landscape import *

def modgradlogpsi(psi, points, eps=0, gpoints=200):
    grid_x, grid_y = np.mgrid[-0.5 : 0.5 : gpoints * 1j, -0.5 : 0.5 : gpoints * 1j]
    psigrid = np.array(griddata(points, np.abs(psi), (grid_x, grid_y)))
    n = gpoints
    logpsi = np.log(psigrid + eps)
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

res_path = data_path
path=os.path.realpath(os.path.join(res_path,"film_poles"))
h = 0.01
gauge = "Sym"
N_eig = 1
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
tri=matplotlib.tri.Triangulation(Th.q[:,0], Th.q[:,1], triangles=Th.me)
print("Mesh done")
N_a = 200
x = 0.15
sigma = 2.2
v = 0
nframes=15
NBmax,NBmin=10,0
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
for num in (1,):
    for NV in (100,):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=[22/2,9/2])
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax1.axis("equal")
        ax2.set_xlim(-0.5,0.5)
        ax2.set_ylim(-0.5,0.5)
        metadata=dict(title='testfilm', artist='Alioune Seye')
        writer= PillowWriter(fps=15, metadata=metadata)
        vmin=10**-1
        vmax=10**1
        vmin_grad=10**-4
        vmax_grad=10**-3
        #plotpsi=ax1.tripcolor(tri, np.ones(Th.nq))
        plotpsi=ax1.tripcolor(Th.q[:,0],Th.q[:,1],Th.me, np.zeros(Th.nq),norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap="turbo",shading="gouraud")
        fig.colorbar(plotpsi,ax=ax1)
        plotgrad=ax2.imshow(np.zeros((10,10)).T, origin="lower", cmap="gist_heat", extent=(-0.5,0.5,-0.5,0.5),vmin=vmin_grad,vmax=vmax_grad)

        psi_frames=[]
        grad_frames=[]
        for frame in range(len(nbs)):
            NB=nbs[frame]


            """
            coeff=1.1
            print("load u")
            nameu = os.path.realpath(
                os.path.join(
                    data_path,
                    "landscapes",
                    "Na400x15sig22v0NV" + str(NV) + "NB" + str(NB) + ".npz",
                )
            )
            u = np.load(nameu, allow_pickle=True)["u"]
            if len(u) != Th.nq:
                print("u does not have th.nq values")
                exit()
            # we make sure we can invert u by adding epsilon
            epsilon = 10**-20
            u = np.abs(u) + epsilon

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

            E=NB**2/2
            def shifted_cond(min, barr):
                return barr - E > coeff * (min - E)


            merged = wsm.merge_algorithm(regions, shifted_cond)
            final_min = 0
            for r in merged.regions:
                if not r.removed:
                    final_min += 1

            print("final number of regions:", final_min)

            x = []
            y = []
            for n in range(Th.nq):
                if len(regions.global_boundary[n]) >= 2:
                    x.append(Th.q[n, 0])
                    y.append(Th.q[n, 1])
            """








            namedata = os.path.realpath(
            os.path.join(
                res_path,
                "film_poles",
                namepot
                +
                "NV"+ str(NV)+ "NBmin"+ str(int(NBmin))+"NBmax"+ str(int(NBmax))+ gauge+ "h"+ str(int(1 / h))+ "Neig"+ str(N_eig)+'frame'+ str(frame)
                + ".npz"
            ))
            psi=np.load(namedata)["eig_vec"][:, num-1]
            psi_frames.append(np.abs(psi))
            grad_frames.append(np.minimum(modgradlogpsi(psi, Th.q, gpoints=400, eps=10**-10), 0.0008))

        def animate(iter):
            fig.suptitle('frame'+str(iter))
            #plotpsi=ax1.tripcolor(Th.q[:,0],Th.q[:,1],Th.me, psi_frames[iter],norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap="turbo")
            plotpsi.set_array(psi_frames[iter])
            
            #plotgrad=ax2.imshow(grad_frames[iter].T, origin="lower", cmap="gist_heat", extent=(-0.5,0.5,-0.5,0.5),vmin=vmin_grad,vmax=vmax_grad)
            plotgrad.set_array(grad_frames[iter].T)
            return plotpsi,plotgrad

        '''#plt.scatter(x, y, c="k", s=1,zorder=2)
        plotpsi.set_data(Th.q[:, 0], Th.q[:, 1], Th.me, np.abs(psi), norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap="gist_heat")
        #cbar1 = plt.colorbar(im1)
        ax1.set_label("$log(modulus(\psi ))$ ")

        plotgrad.set_data(gradlog.T, origin="lower", cmap="turbo", extent=(-0.5,0.5,-0.5,0.5),vmin=vmin_grad,vmax=vmax_grad)
        #plt.scatter(x, y, c="k", s=1)
        #cbar2 = plt.colorbar(im2)
        plotgrad.set_label("$grad(log(modulus(\psi )))$ ")
        #t="gradlogpsi_NV"+str(NV)+"NB"+str(NB) + "num"+str(num)+"boundaries"
                
        with writer.saving(fig, "testfilm.gif", 80):
        
            writer.grab_frame()
                '''
        #animate(5)
        ani= FuncAnimation(fig,animate,frames=nframes,blit=True,repeat=True,interval=1000)
                
        ani.save('filgrad.gif',dpi=100,fps=2)      
        plt.show()
        plt.close()