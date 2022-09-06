import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def PlotValOld(Th,u):
  if Th.d==2:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(Th.q[:,0],Th.q[:,1],u,triangles=Th.me, cmap=plt.cm.Spectral)
    plt.show()
   
# options={"colors":'k',"linewidths":4}
def PlotIsolines(Th,u,**kwargs):
  if isinstance(Th,np.ndarray):
    d=2
  else:
    d=Th.d
  if d==2:
    N=kwargs.get('N', 15 )
    Fill=kwargs.get('fill', False )
    iso=kwargs.get('iso', None )
    Colorbar=kwargs.get('colorbar', False )
    options=kwargs.get('options', None )
    #ColorbarOptions=kwargs.get('ColorbarOptions', {orientation : u'horizontal'} )
    fig = plt.gcf()
    plt.gca().set_aspect('equal')
    if Fill:
      #plt.tricontourf(Th.q[:,0],Th.q[:,1],Th.me, u,N,shading='interp')
      plt.tripcolor(Th.q[:,0],Th.q[:,1],Th.me, u, shading='gouraud')#, cmap=plt.cm.rainbow)
    else:
      #plt.tricontour(Th.q[:,0],Th.q[:,1],Th.me, u,N,colors=coloriso)
      plt.tricontour(Th.q[:,0],Th.q[:,1],Th.me, u,levels=iso,**options)
    if Colorbar:
      #plt.colorbar(orientation=u'horizontal')
      plt.colorbar()
    #plt.show()
  
def PlotMesh(Th,**kwargs):
  assert(Th.d==2)
  color=kwargs.get('color', [0,0,0] )
  fig = plt.gcf()
  plt.triplot(Th.q[:,0],Th.q[:,1],triangles=Th.me,color=color)
  plt.gca().set_aspect('equal')
  plt.show()
    
  
def PlotBounds(Th,**kwargs):
  assert(Th.d==2)
  linewidth=kwargs.get('linewidth', 2.0 )
  legend=kwargs.get('legend', True )
  fontsize=kwargs.get('fontsize', 16 )
  Color=kwargs.get('color', None )
  LB=np.unique(Th.bel)
  fig = plt.gcf()
  plt.rc('text', usetex=True)
  ax = fig.gca()
  ax.axis('equal')
  Lines=[]
  for i in range(len(LB)):
    I,=np.where(Th.bel==LB[i])
    line,=ax.plot(np.r_[Th.q[Th.be[I[0],0],0],Th.q[Th.be[I[0],1],0]],np.r_[Th.q[Th.be[I[0],0],1],Th.q[Th.be[I[0],1],1]],linewidth=linewidth)
      
    Lines.append(line)
    line.set_label(r"$\Gamma_{"+str(int(LB[i]))+"}$")
    
  for i in range(len(LB)):
    I,=np.where(Th.bel==LB[i])
    if Color==None:
      color=Lines[i].get_color()
    else:
      color=Color
   
    for k in I:
      line,=ax.plot(np.r_[Th.q[Th.be[k,0],0],Th.q[Th.be[k,1],0]],np.r_[Th.q[Th.be[k,0],1],Th.q[Th.be[k,1],1]],color=color,linewidth=linewidth)
    
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  #ax.legend(loc='best')
  if legend:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True,fontsize=fontsize)
  #plt.show()
  
  
def PlotVal(Th,x,**kwargs):
  assert(Th.d==2)
  N=kwargs.get('N', None )
  colorbar=kwargs.get('colorbar', True )
  isbounds=kwargs.get('isbounds', True )
  options=kwargs.get('options', None )
  #coloriso=kwargs.get('coloriso', None )
  caxis=kwargs.get('caxis', None )
  linewidth=kwargs.get('linewidth',None)
  #plt.hold(True)
  if isbounds:
    PlotBounds(Th,legend=False,color='k',linewidth=linewidth)  
    
#  plt.axis('off')
  PlotIsolines(Th,x,fill=True,colorbar=colorbar)
  if caxis!=None :
    plt.clim(caxis[0],caxis[1])
  if N!=None:
    PlotIsolines(Th,x,N=N,options=options)
  if caxis!=None :
    plt.clim(caxis[0],caxis[1])
  
def FillMesh(Th,**kwargs):
  # Color 
  #   Gray shades can be given as a string encoding a float in the 0-1 range : Color='0.2'
  #   R , G , B tuple, where each of R , G , B are in the range [0,1] : Color=(0.2,0.1,0.1)
  assert(Th.d==2)
  Color=kwargs.get('Color', '0.9' ) # Gray shades can be given as a string encoding a float in the 0-1 range
  fig = plt.gcf()
  ax = fig.gca()
  for i in range(Th.nme):
    ax.add_patch(Polygon(Th.q[Th.me[i]], closed=True,color=Color))
  #plt.show()
  
def PlotEig(pde,eigenvector,eigenvalue,**kwargs):
  comp=kwargs.get('comp',None)
  caxis=kwargs.get('caxis', None )
  titleoptions=kwargs.get('titleoptions', None )
  niso=kwargs.get('niso', 0 )
  plt.hold(True)
  plt.axis('off') 
  if max(abs(eigenvector.imag)) < 1e-12 :
    u=eigenvector.real
  else:
    u=abs(eigenvector)
  PlotVal(pde.Th,u,isbounds=False,colorbar=False,caxis=caxis)#,options=options)
  #if caxis!=None :
    #plt.clim(caxis[0],caxis[1])
  PlotIsolines(pde.Th,u,iso=[0],options={"colors":'k',"linewidths":3})
  if niso>0 :
    iso=Tchebycheff(min(u),max(u),niso)
    PlotIsolines(pde.Th,u,iso=iso,options={"colors":'k',"linewidths":1})
  plt.rc('text', usetex=True)
  if abs(eigenvalue.imag) < 1e-12 :
    plt.title(r'$\lambda=%.3f$'%eigenvalue.real,**titleoptions)
  else:
    plt.title(r'$\lambda=(%.3f,%.3f)$'%(eigenvalue.real,eigenvalue.imag),**titleoptions)
  plt.hold(False)
  
def PlotEigs(pde,eigenvectors,eigenvalues,**kwargs):
  comp=kwargs.get('comp',None)
  caxis=kwargs.get('caxis', None )
  niso=kwargs.get('niso', 0 )
  titleoptions=kwargs.get('titleoptions', None )
  NumEigs=eigenvalues.shape[0]
  plt.close("all")

  plt.ion() # interactive mode
  for i in range(NumEigs):
    plt.figure(i)
    plt.clf()
    plt.hold(True)
  
    #PlotBounds(Th,legend=False,color='k')
    plt.axis('off') 
    options={"colors":'k',"linewidths":1}
    if max(abs(eigenvectors[::,0].imag)) < 1e-12 :
      u=eigenvectors[::,i].real
    else:
      u=abs(eigenvectors[::,i])
    PlotVal(pde.Th,u,isbounds=False,colorbar=False,caxis=caxis)#,options=options)
    #if caxis!=None :
      #plt.clim(caxis[0],caxis[1])
    PlotIsolines(pde.Th,u,iso=[0],options={"colors":'k',"linewidths":3})
    if niso>0 :
      iso=Tchebycheff(min(u),max(u),niso)
      PlotIsolines(pde.Th,u,iso=iso,options={"colors":'k',"linewidths":1})
    plt.rc('text', usetex=True)
    if abs(eigenvalues[i].imag) < 1e-12 :
      plt.title(r'$\lambda=%.3f$'%eigenvalues[i].real,**titleoptions)
    else:
      plt.title(r'$\lambda=(%.3f,%.3f)$'%(eigenvalues[i].real,eigenvalues[i].imag),**titleoptions)
    plt.hold(False)

def Tchebycheff(a,b,n):
  return 0.5*((a+b)+(b-a)*np.cos(np.pi+(2*np.arange(n+1)+1)*np.pi/(2*(n+1))))
  
def showSparsity(M):
#  from matplotlib.pyplot as plt
  plt.spy(M, precision=1e-8, marker='.', markersize=3)
  plt.show()
  
def SaveFigAsFile(nfig,savefile,**kwargs):
  #options=kwargs.get('options', None )
  fig=plt.figure(nfig)
  fig.set_rasterized(True)
  plt.gca().set_axis_off()
  #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
  #                    hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(savefile,bbox_inches='tight', pad_inches=0,**kwargs)