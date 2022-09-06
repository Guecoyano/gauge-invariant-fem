import numpy as np
import types

class Loperator:
  def __init__(self,**kwargs):
    self.d=kwargs.get('d', 2)
    self.m=1
    type=kwargs.get('type', None)
    self.name=kwargs.get('name', 'No name')
    self.order=0;
    if (type=='fill'):
      self.A =kwargs.get('A', NoneMatrix(self.d,self.d))
      self.b =kwargs.get('b', NoneVector(self.d))
      self.c =kwargs.get('c', NoneVector(self.d))
    else:
      self.A =kwargs.get('A', None)
      self.b =kwargs.get('b', None)
      self.c =kwargs.get('c', None)
    self.a0=kwargs.get('a0', None)
    if self.A!=None:
      self.order=2
      for i in range(self.d):
        for j in range(self.d):
          if isinstance(self.A[i][j],types.FunctionType):
            self.A[i][j]=np.vectorize(self.A[i][j])
    if self.b!=None  :
      self.order=max(self.order,1)
      for i in range(self.d):
        if isinstance(self.b[i],types.FunctionType):
          self.b[i]=np.vectorize(self.b[i])
    if self.c!=None : 
      self.order=max(self.order,1)
      for i in range(self.d):
        if isinstance(self.c[i],types.FunctionType):
          self.c[i]=np.vectorize(self.c[i])
    if isinstance(self.a0,types.FunctionType):
      self.order=max(self.order,0) 
      self.a0=np.vectorize(self.a0)
      
  def __repr__(self):
    return "Loperator %s"%(self.name)
    
  def __str__(self,*args):      
    return "member of Test"

class Hoperator:
  def __init__(self,**kwargs):
    self.d=kwargs.get('d', 2)
    self.m=kwargs.get('m', 2)
    self.H=kwargs.get('H', NoneMatrix(self.m,self.m))
    self.name=kwargs.get('name', 'No name')
    self.params=kwargs.get('params', None)
    self.order=0
    
def isoperatorL(operator):
  if isinstance(operator,Loperator):
    return True
  return False
  
def isoperatorH(operator):
  if isinstance(operator,Hoperator):
    return True
  return False  

def setDorder(Dop):
  if (Dop.A!=None):
    for i in range(Dop.d):
      for j in range(Dop.d):
        if (Dop.A[i][j]!=None) :
         return 2
  if (Dop.b!=None):
    for i in range(Dop.d):
      if (Dop.b[i]!=None) :
        return 1
  if (Dop.c!=None):
    for i in range(Dop.d):
      if (Dop.c[i]!=None) :
        return 1
  return 0

def setHorder(Hop):
  order=0;
  for i in range(Hop.m):
    for j in range(Hop.m):
      if (Hop.H[i][j]!=None) :
        order=max(order,Hop.H[i][j].order)
  return order

def NoneVector(d):
  V=[]
  for i in range(d):
    V.append(None)
  return V
  
def NoneMatrix(m,n):
  M=[]
  for i in range(m):
    M.append(NoneVector(n))
  return M


# Operateurs predefinis
#######################

def Lmass(d,a0):
  return Loperator(d=d,a0=a0,name='MassW')
  
def Lstiff(d):
  A=NoneMatrix(d,d)
  for i in range(d):
    A[i][i]=1.
  return Loperator(d=d,A=A,name='Stiff')
    
def StiffElasHoperators(d,lam,mu):
  assert(d==2 or d==3)
  Hop=Hoperator(m=d,d=d,name='StiffElas',params={'lam':lam,'mu':mu})
  Hop.order=2;
  gam=None
  if np.isscalar(lam) and np.isscalar(mu):
    gam=lam+2*mu
  if isinstance(lam, type(lambda: None)) and np.isscalar(mu):
    if (d==2): gam=lambda x1,x2 :lam(x1,x2) + 2*mu
    if (d==3): gam=lambda x1,x2,x3 :lam(x1,x2,x3) + 2*mu
  if isinstance(mu, type(lambda: None)) and np.isscalar(lam):
    if (d==2): gam=lambda x1,x2 :lam + 2*mu(x1,x2)
    if (d==3): gam=lambda x1,x2,x3 :lam + 2*mu(x1,x2,x3)
  if isinstance(mu, type(lambda: None)) and isinstance(lam, type(lambda: None)):
    if (d==2): gam=lambda x1,x2 :lam(x1,x2) + 2*mu(x1,x2)
    if (d==3): gam=lambda x1,x2,x3 :lam(x1,x2,x3) + 2*mu(x1,x2,x3)
  assert(gam!=None)
    
  if d==2:
    Hop.H[0][0]=Loperator(d=d,A=[[gam,None],[None,mu]]) 
    Hop.H[0][1]=Loperator(d=d,A=[[None,lam],[mu,None]]) 
    Hop.H[1][0]=Loperator(d=d,A=[[None,mu],[lam,None]]) 
    Hop.H[1][1]=Loperator(d=d,A=[[mu,None],[None,gam]])
  if d==3:
    Hop.H[0][0]=Loperator(d=d,A=[[gam,None,None],[None,mu,None],[None,None,mu]]) 
    Hop.H[0][1]=Loperator(d=d,A=[[None,lam,None],[mu,None,None],[None,None,None]]) 
    Hop.H[0][2]=Loperator(d=d,A=[[None,None,lam],[None,None,None],[mu,None,None]]) 
    Hop.H[1][0]=Loperator(d=d,A=[[None,mu,None],[lam,None,None],[None,None,None]]) 
    Hop.H[1][1]=Loperator(d=d,A=[[mu,None,None],[None,gam,None],[None,None,mu]])
    Hop.H[1][2]=Loperator(d=d,A=[[None,None,None],[None,None,lam],[None,mu,None]])
    Hop.H[2][0]=Loperator(d=d,A=[[None,None,mu],[None,None,None],[lam,None,None]])
    Hop.H[2][1]=Loperator(d=d,A=[[None,None,None],[None,None,mu],[None,lam,None]])
    Hop.H[2][2]=Loperator(d=d,A=[[mu,None,None],[None,mu,None],[None,None,gam]])
  return Hop
