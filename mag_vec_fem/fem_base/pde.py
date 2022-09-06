from fem_base import *
from fem_base import mesh
from fem_base import FEM
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve,gmres,bicgstab,cg,minres,splu,qmr
import time

# replace initPDE in algo
class PDE:
  def __init__(self,Op,Th,**kwargs):
    assert( (isinstance(Op,operators.Loperator) or isinstance(Op,operators.Hoperator)) and isinstance(Th,mesh.Mesh))
    assert(Op.d==Th.d)
    self.dtype=kwargs.get('dtype', float)
    self.d=Th.d
    self.m=Op.m
    self.Th=Th
    self.op=Op
    if Op.m>1:
      self.f=operators.NoneVector(Op.m)
    else:
      self.f=None
    self.Bh=mesh.BuildBoundaryMeshes(Th)
    self.labels=np.unique(Th.bel)
    self.nlab=self.labels.shape[0]
    self.bclD=operators.NoneMatrix(Op.m,self.nlab)
    self.bclR=operators.NoneMatrix(Op.m,self.nlab)
    self.uex=None;
 
# for compatibility with report
def initPDE(Op,Th):
  return PDE(Op,Th)

# replace setBC in algo    
class BC:
  def __init__(self,ctype,g,ar):
    self.type=ctype
    self.g=g
    self.ar=ar
    if ctype=='Dirichlet':
      self.id=0
    elif ctype=='Neumann':
      self.id=1
    elif ctype=='Robin':
      self.id=2
    else:
      print("Unknow BC -> set to homogeneous Neumann")
      self.type='Neumann'
      self.id=1
      self.g=self.ar=0
  
class BCdirichlet:
  def __init__(self,g):
    self.g=g

class BCrobin:
  def __init__(self,g,ar):
    self.g=g
    self.ar=ar
    
def getFunc(gfuncs,i):
  if gfuncs==None:
    return None
  if isinstance(gfuncs, list):
    return gfuncs[i]
  return gfuncs
    
def setBC_PDE(pde,label,comps,ctype,gfuncs,arfuncs=None):
  l=np.where(pde.labels==label)[0]
  if l.shape[0]==0:
    print("label %d not found in mesh!" %label)
  nc=1
  if isinstance(comps,int):
    comps=[comps]
  #if isinstance(comps,list):  
  nc=len(comps)
  if ctype=='Dirichlet':
    for i in range(nc):
      for j in l:
        pde.bclD[comps[i]][j]=BCdirichlet(getFunc(gfuncs,i))
  if ctype=='Neumann':
    for i in range(nc):
      for j in l:
        pde.bclR[comps[i]][j]=BCrobin(getFunc(gfuncs,i),None)
  if ctype=='Robin':
    for i in range(nc):
      for j in l:
        pde.bclR[comps[i]][j]=BCrobin(getFunc(gfuncs,i),getFunc(arfuncs,i))    
  return pde
  
def RHS(Th,f,Num,**kwargs):
  dtype=kwargs.get('dtype', float)
  version=kwargs.get('version','OptV3')
  ff=f if isinstance(f, list) else [f]
  m=len(ff)
  VFInd=FEMtools.getVFindices(Num,m,Th.nq)
  F=np.zeros((m*Th.nq,),dtype=dtype)
  LMass=operators.Lmass(Th.d,1)
  Mass=FEM.AssemblyP1(Th,LMass,version=version)  
  I=np.arange(Th.nq)
  for i in range(m):
    if ff[i]!=None:
      V=Mass*FEMtools.setFdata(ff[i],Th,dtype=dtype)
      F[VFInd(I,i)]=Mass*FEMtools.setFdata(ff[i],Th,dtype=dtype)
  return F

def RobinBC(pde,AssemblyVersion,Num):
  m=pde.m;nq=pde.Th.nq
  VFNum=FEMtools.getVFindices(Num,m,nq)
  FR=np.zeros((m*nq,))
  II=np.array([],dtype=int)
  JJ=np.array([],dtype=int)
  KK=np.array([],dtype=pde.dtype)
  LMass=operators.Loperator(d=pde.Th.d,a0=1.0)
  for l in range(pde.nlab):
    MassBC=None
    for i in range(m):
      Ind=VFNum(pde.Bh[l].toGlobal,i)
      if pde.bclR[i][l]!=None:
        if  pde.bclR[i][l].g!=None:
          if (MassBC==None):
            MassBC=FEM.AssemblyP1(pde.Bh[l],LMass,version=AssemblyVersion)
          g=FEMtools.setFdata(pde.bclR[i][l].g,pde.Bh[l],dtype=pde.dtype)
          FR[Ind]+=MassBC*g
        if  pde.bclR[i][l].ar!=None:
          Kg=FEMOptV3.KgP1_OptV3_guv(pde.Bh[l],pde.bclR[i][l].ar,dtype=pde.dtype)
          Ig,Jg=FEMOptV3.IgJgP1_OptV3(pde.Bh[l].d,pde.Bh[l].nme,pde.Bh[l].me)
          Ig=Ind[Ig]
          Jg=Ind[Jg]
          II=np.concatenate((II,np.reshape(Ig,(np.prod(Ig.shape),))))
          JJ=np.concatenate((JJ,np.reshape(Jg,(np.prod(Jg.shape),))))
          KK=np.concatenate((KK,np.reshape(Kg,(np.prod(Kg.shape),))))
  MR=sparse.csc_matrix((KK,(II,JJ)),shape=(m*nq,m*nq))
  return MR,FR
        
def DirichletBC(pde,Num):    
  m=pde.m;nq=pde.Th.nq;
  VFNum=FEMtools.getVFindices(Num,m,nq)
  ndof=m*nq
  g=np.zeros((ndof,),dtype=pde.dtype)
  IndD=np.array([],dtype=int)
  for l in range(pde.nlab):
    for i in range(m):
      if pde.bclD[i][l]!=None:
        Ind=VFNum(pde.Bh[l].toGlobal,i)
        g[Ind]=FEMtools.setFdata(pde.bclD[i][l].g,pde.Bh[l],dtype=pde.dtype)
        IndD=np.concatenate((IndD,Ind))
  ID=np.unique(IndD)
  IDc=np.setdiff1d(np.arange(ndof),ID)
  return ID,IDc,g
  
class SolveInfo():
    def __init__(self,AssemblyVersion,SolveVersion,SolveOptions,tcpu,flag,residu,ndof):
      self.AssemblyVersion=AssemblyVersion
      self.SolveVersion=SolveVersion
      self.SolveOptions=SolveOptions
      self.tcpu=tcpu
      self.flag=flag
      self.residu=residu
      self.ndof=ndof
    def __str__(self,*args):
      str1 = "     Assembly version : %s\n     Solve version    : %s\n     Solve options    : %s\n"%(self.AssemblyVersion,self.SolveVersion,self.SolveOptions)
      str1 += "     matrix size=%d\n"%self.ndof
      str1 += "     residu=%.16e, flag=%d\n"%(self.residu,self.flag)
      str1 += "     Assembly              : %.4f(s)\n"%self.tcpu[0]
      str1 += "     RHS                   : %.4f(s)\n"%self.tcpu[1]
      str1 += "     Boundary Conditions : %.4f(s)\n"%self.tcpu[2]
      str1 += "     Solve                 : %.4f(s)\n"%self.tcpu[3]
      return str1
    
def buildPDEsystem(pde,**kwargs):
  Num=kwargs.get('Num',1)
  AssemblyVersion=kwargs.get('AssemblyVersion','OptV3')
  M=FEM.AssemblyP1(pde.Th,pde.op,Num=Num,dtype=pde.dtype,version=AssemblyVersion)
  ndof=M.get_shape()[0]
  b=RHS(pde.Th,pde.f,Num,dtype=pde.dtype,version=AssemblyVersion)
  [AR,bR]=RobinBC(pde,AssemblyVersion,Num)
  A=M+AR
  b=b+bR
  [ID,IDc,gD]=DirichletBC(pde,Num)
  return A,b,ID,IDc,gD
    
def solvePDE(pde,**kwargs):
  Num=kwargs.get('Num',1)
  AssemblyVersion=kwargs.get('AssemblyVersion','OptV3')
  SolveVersion=kwargs.get('SolveVersion','classic')
  SolveOptions=kwargs.get('SolveOptions',None)
  verbose=kwargs.get('verbose',False)
  split=kwargs.get('split',False)
  Tcpu=np.zeros((4,))
  tstart=time.time()
  M=FEM.AssemblyP1(pde.Th,pde.op,Num=Num,dtype=pde.dtype,version=AssemblyVersion)
  Tcpu[0]=time.time()-tstart
  ndof=M.get_shape()[0]
  tstart=time.time()
  b=RHS(pde.Th,pde.f,Num,dtype=pde.dtype,version=AssemblyVersion)
  Tcpu[1]=time.time()-tstart
  tstart=time.time()
  #bN=NeumannBC(pde,AssemblyVersion,Num);
  [AR,bR]=RobinBC(pde,AssemblyVersion,Num)
  b=b+bR#+bN
  [ID,IDc,gD]=DirichletBC(pde,Num)
  A=M+AR;
  #b[IDc]=b[IDc]-A[IDc,::]*gD;
  Tcpu[2]=time.time()-tstart
  x=np.zeros((ndof,),dtype=pde.dtype)
  tstart=time.time()
  X,flag=globals()[SolveVersion+"Solve"](A,b,ndof,gD,ID,IDc,pde.dtype,SolveOptions)
  Tcpu[3]=time.time()-tstart
  if split:
    x=splitPDEsol(pde,X,Num) 
  else: 
    x=X
  if (verbose):
    bb=b[IDc]-A[IDc,::]*gD;
    residu=np.linalg.norm(A[IDc][::,IDc]*X[IDc]-bb);
    Nh=np.linalg.norm(b[IDc])
    if (Nh>1e-9):
      residu/=Nh;
    return x,SolveInfo(AssemblyVersion,SolveVersion,SolveOptions,Tcpu,flag,residu,ndof)
  else:
    return x
  
def splitPDEsol(pde,X,Num):
  if pde.m==1:
    return X
  VFInd=FEMtools.getVFindices(Num,pde.m,pde.Th.nq)
  x=operators.NoneVector(pde.m)
  I=np.arange(pde.Th.nq)
  for i in range(pde.m):
    x[i]=X[VFInd(I,i)]
  return x

def splitSol(X,m,nq,**kwargs):
  Num=kwargs.get('Num',1)
  if m==1:
    return X
  VFInd=FEMtools.getVFindices(Num,m,nq)
  x=operators.NoneVector(m)
  I=np.arange(nq)
  for i in range(m):
    x[i]=X[VFInd(I,i)]
  return x

def getSolveVersions():
  return ['classic','gmres','bicgstab','cg','minres','spLU','qmr']
  
def classicSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc]=spsolve((A[IDc])[::,IDc],bb)
  flag=0
  return x,flag

# SolveOptions = {"x0": None, "tol": 1e-05, "restart":None, "maxiter":None}
def gmresSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc],flag=gmres((A[IDc])[::,IDc],bb,**SolveOptions)
  return x,flag

# SolveOptions = {"x0": None, "tol": 1e-05, "maxiter":None}
def bicgstabSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc],flag=bicgstab((A[IDc])[::,IDc],bb,**SolveOptions)
  return x,flag

# SolveOptions = {"x0": None, "tol": 1e-05, "maxiter":None}
def cgSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc],flag=cg((A[IDc])[::,IDc],bb,**SolveOptions)
  return x,flag

# SolveOptions = {"x0": None, "tol": 1e-05, "maxiter":None}
def minresSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc],flag=minres((A[IDc])[::,IDc],bb,**SolveOptions)
  return x,flag

# Use SuperLU
def spLUSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  lu=splu((A[IDc])[::,IDc])
  x[IDc]=lu.solve(bb)
  return x,0

# SolveOptions = {"x0": None, "tol": 1e-05, "maxiter":None}
def qmrSolve(A,b,ndof,gD,ID,IDc,dtype,SolveOptions):
  x=np.zeros((ndof,),dtype=dtype)
  x[ID]=gD[ID]
  bb=b[IDc]-A[IDc,::]*gD;
  x[IDc],flag=qmr((A[IDc])[::,IDc],bb,**SolveOptions)
  return x,flag


        
