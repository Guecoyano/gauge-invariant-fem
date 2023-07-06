import numpy as np
import matplotlib.pyplot as plt
from fem_base.mesh import HyperCube
d=2
N=6
L=5
Th=HyperCube(d,N,l=L)
x=np.linspace(-L/2,L/2,N)
y=np.linspace(-L/2,L/2,N)
#in Th.q x changes fastest so to cast Ay to Th.nq shape, we have to reshape.
A_x=np.repeat(-y/2,N)
A_y=np.ravel(np.reshape(np.repeat(x/2,N),(N,N)),'F')

def circultion_A(A_x,A_y,Th): #takes Ax and Ay as arrays of size Th.nq and returns an array of circulation of A along edges of Th of size (Th.nme,Th.d+1,Th.d+1)
    A_xme=A_x[Th.me]
    A_yme=A_y[Th.me]
    qme=Th.q[Th.me]
    computed_circ=np.zeros((Th.nme,Th.d+1,Th.d+1))
    for i in range(Th.d+1):
        for j in range(i):
            computed_circ[:,i,j]=(qme[:,j,0]-qme[:,i,0])*(A_xme[:,i]+A_xme[:,j])/2+(qme[:,j,1]-qme[:,i,1])*(A_yme[:,i]+A_yme[:,j])/2
            computed_circ[:,j,i]=-computed_circ[:,i,j]
    return computed_circ



def plotmesh(Th):
    fig=plt.figure()
    qme=Th.q[Th.me]
    for k in range(Th.nme):
        for i in range(3):
            for j in range(i):
                plt.plot([qme[k,i,0],qme[k,j,0]],[qme[k,i,1],qme[k,j,1]])
    plt.show()
    plt.clf()
    plt.close()