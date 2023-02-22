# coding=utf-8
from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi

import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
from fem_base.graphics import PlotVal, PlotMesh, PlotBounds, FillMesh
from matplotlib.text import Text
from math import cos
from scipy.interpolate import griddata
import numpy as np


"""Test derived from the sample poisson2D01"""


plt.close("all")

d = 2
# N=int(input("N="))
l = 10**-7
N = 5
N1 = 11  # int(l/(5.6*10**-10))
B = 10
h = 0.05
V_max = 0.5 * 1.6 * 10**-19
V_maxmeV = 50
v = 1
x = 0.2
Th = HyperCube(2, int(1 / h), l=l)
vgauss, points = gi.gauss_pot(V_max, 2.3, N1, l, x)
V = gi.interpolate_pot(vgauss, points, Th.q)

# V=lambda x,y:abs(x+l/2)*10**-13
lnm = 100
nameV = "pre_interp_potv" + str(v) + "l" + str(lnm) + "E1eVx15.npy"
nameq = "Vql" + str(lnm) + ".npy"
nameTh = "Th" + str(int(1 / h)) + "l" + str(lnm)
# Th=np.load(nameTh)
# V_preinterp,points=0.001*V_maxmeV*np.load(nameV),np.load(nameq)
# V=gi.interpolate_pot(V_preinterp,points,Th.q)

print("Get first 5 eigenfunctions : 2D Magnetic Schrödinger")
w, x = gi.get_eigvv(N=N, B=B, h=h, gauge="LandauX", V=V, Th=Th)
w2, x2 = gi.get_eigvv(N=N, B=B, h=h, gauge="Sym", V=V, Th=Th)

print("4. Post-processing")
print(w, w2)
plt.figure(1)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x[:, 0]))
plt.title(
    r"2D Magnetic Schrödinger : first eigenfunction (modulus), $E_1=%d$)"
    % (w[0])
)


plt.figure(2)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x[:, 2]))
plt.title(
    r"2D Magnetic Schrödinger : third eigenfunction (modulus), $E_3=%d$)"
    % (w[2])
)


plt.figure(3)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x[:, 4]))
plt.title(
    r"2D Magnetic Schrödinger : fifth eigenfunction (modulus), $E_5=%d$)"
    % (w[4])
)


plt.figure(4)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x2[:, 0]))
plt.title(
    r"2D Magnetic Schrödinger : first eigenfunction (modulus), $E_1=%d$)"
    % (w2[0])
)


plt.figure(5)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x2[:, 2]))
plt.title(
    r"2D Magnetic Schrödinger : third eigenfunction (modulus), $E_3=%d$)"
    % (w2[2])
)


plt.figure(6)
plt.clf()
PlotBounds(Th, legend=False, color="k")
plt.axis("off")
PlotVal(Th, np.abs(x2[:, 4]))
plt.title(
    r"2D Magnetic Schrödinger : fifth eigenfunction (modulus), $E_5=%d$)"
    % (w2[4])
)
plt.show()
plt.close()
