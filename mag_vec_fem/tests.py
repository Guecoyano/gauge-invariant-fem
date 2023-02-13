# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
from os import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

'''print(sys.path)
pot_version=0
lnm=200
h=0.05
gauge='Sym'
l=lnm*10**-9
N_eig=100
res_path=data_path'''
def connectivity(Th):
    """Build connectivity array from mesh Th. Th.q is the list of vertices coordinates, Th.me is the list of mesh elements.
    Th.me[b,k] is the index of vertex b in the k-th mesh element.
    """
    conn=Th.nq*[[]]
    for me in Th.me:
        if not me[0] in conn[me[1]]:
            conn[me[0]]=conn[me[0]]+[me[1]]
            conn[me[1]]=conn[me[1]]+[me[0]]
        if not me[0] in conn[me[2]]:
            conn[me[0]]=conn[me[0]]+[me[2]]
            conn[me[2]]=conn[me[2]]+[me[0]]
        if not me[2] in conn[me[1]]:
            conn[me[2]]=conn[me[2]]+[me[1]]
            conn[me[1]]=conn[me[1]]+[me[2]]
    return conn
Th=HyperCube(2,3,l=1)
print(Th.q.shape)
print(Th.nme)
print(Th.me)
conn=connectivity(Th)
print(conn)