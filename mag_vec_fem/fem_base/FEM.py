import fem_base.operators
from fem_base.FEMOptV3 import *
from fem_base.FEMtools import getVFindices

import numpy as np
from scipy import linalg
from scipy import sparse
import itertools
import types
from math import *
  
def  AssemblyP1(Th,Op,**kwargs):
  version=kwargs.get('version', 'OptV3') 
  gradient=kwargs.get('gradient', None) 
  Num=kwargs.get('Num',1)
  #dtype=kwargs.get('dtype',float)
  if operators.isoperatorL(Op):
    return globals()['DAssemblyP1_'+version](Th,Op,gradient)
  if operators.isoperatorH(Op):
    return globals()['HAssemblyP1_'+version](Th,Op,Num)


def NormH1(Th,U,**kwargs):
  Num=kwargs.get('Num', 1)
  m=len(U)//Th.nq  # division entiere
  if ( m*Th.nq != len(U) ):
    print('dimension error m=%d,nq=%d,len(U)=%d'%(m,Th.nq,len(U)))
  VFNum=getVFindices(Num,m,Th.nq)  
  OpM=operators.Lmass(Th.d,1.0)
  OpS=operators.Lstiff(Th.d)
  M=AssemblyP1(Th,OpM)
  K=AssemblyP1(Th,OpS)
  S=0.
  I=np.arange(Th.nq)
  for i in range(m):
    UI=np.abs(U[VFNum(I,i)])
    S+=np.dot(M*UI,UI)+np.dot(K*UI,UI)
  return np.sqrt(S),M,K

def NormL2(Th,U,**kwargs):
  Num=kwargs.get('Num', 1)
  M=kwargs.get('Mass', None)
  if M!=None:
    assert(isinstance(M,sparse.csc.csc_matrix))
    assert(M.get_shape()==(Th.nq,Th.nq))
  else:
    OpM=operators.Lmass(Th.d,1.0)
    M=AssemblyP1(Th,OpM)
  m=round(U.shape[0]/Th.nq)
  assert(m*Th.nq==U.shape[0])
  VFNum=getVFindices(Num,m,Th.nq)
  S=0.
  I=np.arange(Th.nq)
  for i in range(m):
    UI=np.abs(U[VFNum(I,i)])
    S+=np.dot(M*UI,UI)
  return np.sqrt(S),M
