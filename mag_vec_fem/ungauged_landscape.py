from itertools import filterfalse
import fem_base.gaugeInvariantFEM as gi

import matplotlib.pyplot as plt

from fem_base.FEM import *
from fem_base.mesh import *
from fem_base.pde import *
from fem_base.common import *
from fem_base.graphics import PlotVal,PlotMesh,PlotBounds,FillMesh
from matplotlib.text import Text
from math import cos
import numpy as np


'''Test derived from the sample poisson2D01'''


plt.close("all")

d=2
#N=int(input("N="))
N=25
B=1
print("1. Set square mesh")
Th=HyperCube(d,N)
print("  -> Mesh sizes : nq=%d, nme=%d, nbe=%d" % (Th.nq,Th.nme,Th.nbe));

print("2. 3. Set and solve BVP : 2D Magnetic Schrödinger")
x=gi.getSol(Th=Th,B=B)

print("4. Post-processing")

plt.figure(1)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,x.real)
plt.title(r'2D Magnetic Schrödinger : solution (real part) (mesh $n_q=%d$, $n_{me}=%d$)'%(Th.nq,Th.nme))


plt.figure(2)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,x.imag)
plt.title(r'2D Magnetic Schrödinger : solution (imaginary part) (mesh $n_q=%d$, $n_{me}=%d$)'%(Th.nq,Th.nme))


plt.figure(3)
plt.clf()
PlotBounds(Th,legend=False,color='k')
plt.axis('off')
PlotVal(Th,np.abs(x))
plt.title(r'2D Magnetic Schrödinger : solution (modulus) (mesh $n_q=%d$, $n_{me}=%d$)'%(Th.nq,Th.nme))
plt.show()
plt.close()