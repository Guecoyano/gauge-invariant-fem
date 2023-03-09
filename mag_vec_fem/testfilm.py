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





res_path = data_path

N_a = 200
x = 0.15
sigma = 2.2
v = 0
nframes=150
NBmax,NBmin=10,0
nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))
fig=plt.figure()
plt.close("all")
metadata=dict(title='testfilm', artist='Alioune Seye')
writer= PillowWriter(fps=15, metadata=metadata)
fig, (ax1,ax2)= plt.subplots(1,2)
X,Y1,Y2=[],[],[]
ax1.set_xlim(0,NBmax)
ax1.set_ylim(-1,1)
ax2.set_xlim(0,NBmax)
ax2.set_ylim(-1,1)
l1,=ax1.plot([],[])
l2,=ax2.plot(X,Y2)
with writer.saving(fig, "testfilm.gif", 100):
    for frame in range(nframes):
        print(frame)
        X.append(frame*NBmax/nframes)
        Y1.append(np.sin(frame/nframes))
        Y2.append(np.sin(2*frame/nframes))
        #plt.scatter(x, y, c="k", s=1,zorder=2)
        l1.set_data(X,Y1)
        l2.set_data(X,Y2)
        writer.grab_frame()
        
        
        
plt.close()