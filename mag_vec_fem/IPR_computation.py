# coding=utf-8
"""This module defines functions used to compute efficiently the IPRs of eigenvectors of the first landau level for given mesh size, potential, disorder and magnetic parameters."""

import numpy as np
from fem_base.gaugeInvariantFEM import *

def create_m0(Th, IgJg=None):
    Kg = Kg_guv_ml(Th, 1, complex)
    if IgJg is None:
        Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    else:
        Ig,Jg= IgJg
    NN = Th.nme * (Th.d + 1) ** 2
    M = sparse.csc_matrix(
        (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
        shape=(Th.nq, Th.nq),
    )
    M.eliminate_zeros()
    return M

def integrate_mesh(Th,f_q):
    m_0=np.sum(create_m0(Th),0)
    #f_mesh=f_q[Th.me]
    return np.sum(np.dot(f_q,m_0))

def PR(Th,eig_vec):
    m0=np.sum(create_m0(Th),0)
    psi4=np.sum(np.dot(m0,eig_vec**4))
    psi2=np.sum(np.dot(m0,eig_vec**2))
    return psi2**2/psi4