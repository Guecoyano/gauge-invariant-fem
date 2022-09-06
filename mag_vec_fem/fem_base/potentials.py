import numpy as np
rnd=np.random.random
binom=np.random.binomial
from math import *
from scipy.interpolate import griddata

def nrandgaussi( N, deltax, deltay, force):
    for k in range(N):
        ampl,r2,r3=rnd()*force,rnd(),rnd()
        x=deltax*(1/2-r2)
        y=deltay*(1/2-r3)
        print("+")
        print(str(ampl)+"*gaussXY["+str(x)+","+str(y)+"]")


def ngaussi( N, deltax, deltay):
    for k in range(N):
        r2,r3=rnd(),rnd()
        x=deltax*(1/2-r2)
        y=deltay*(1/2-r3)
        print("+")
        print(str(1)+"*gaussXY["+str(x)+","+str(y)+"]")

def alloytypestring(N, x):
    for k in range(N):
        for j in range(N):
            d=binom(21, x)
            print("+")
            print(str(d)+"*gaussXY["+str(-1/2+k/N)+","+str(-1/2+j/N)+"]")

def alloymatrix(N,V, x):
    M=np.zeros((N,N))
    for k in range(N):
        for j in range(N):
            M[k,j]=binom(V, x)/V
    return M
    
def boolmatrix(N, x):
    M=np.zeros((N,N))
    for k in range(N):
        for j in range(N):
            M[k,j]=binom(1, x)
    return M
    
def condbandfrombool(N,x,sigma):
    M0=boolmatrix(N,x)
    M=np.zeros((N,N))
    t=0.
    s=int(sigma//1)
    for l in range (-s,s):
        l1=int(sqrt(sigma**2-l**2)//1)
        for m in range(-l1,l1):
            t+=1
            for k in range(N):
                for j in range(N):
                    if k+l in range(N) and j+m in range(N):
                        M[k,j]+=M0[k+l,j+m]
                    else:
                        M[k,j]+=x
    return M*(1/t)
        

def potential(l,a,sigma,ampl,x,name):
    """
    creates a potential from alloy disorder in a .dat file
    :param l: size of the box in nm
    :param a: crystal mesh parameter in nm
    :param sigma: smearing length
    :param ampl: energy difference in conduction band between the two atomic crystals in meV
    :param x: alloy proportion in percent
    :name: name of the file
    """
    N=int(l//a)
    M=condbandfrombool(N,x,sigma)
    nname=name+str(N)+str(10*sigma)+str(ampl)+str(x)+'.dat'
    with open(nname,'wb') as f:
        f.write(b"      str(N)    str(N) \n")
        for i in range(N):
            for j in range(N):
                f.write(b""+ str(M[i,j])+"     ")
            f.write(b"\n")
    
def gauss_pot(V,sigma,N,L,x):
    # make the point value matrix
    M0=np.random.binomial(1, x, (N,N))
    M=np.zeros((N,N))
    t=0.
    s=int(sigma//1)
    for l in range (-3*s,3*s):
        l1=int(sqrt(9*sigma**2-l**2)//1)
        for m in range(-l1,l1):
            t+=exp(-(l**2+m**2)/(2*sigma**2))
            for k in range(N):
                for j in range(N):
                    if k+l in range(N) and j+m in range(N):
                        M[k,j]+=exp(-(l**2+m**2)/(2*sigma**2))*V*M0[k+l,j+m]
                    else:
                        M[k,j]+=exp(-(l**2+m**2)/(2*sigma**2))*V*(np.random.binomial(1,x)+x)/2
    # make the points coordinates matrix
    n_bins =  (N)*np.ones(2)
    bounds = np.repeat([(-L/2,L/2)], 2, axis = 0)
    A = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]
    q=np.array([A[i].ravel() for i in range(2)]).T
    q=q[:,range(2-1,-1,-1)]
    return M*(1/t),q
  
def interpolate_pot(value_grid,points,q):
  values=np.ravel(value_grid)
  V=griddata(points,values,q)
  return V

