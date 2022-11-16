# coding=utf-8
import numpy as np
from fem_base.potentials import gauss_pot
from fem_base.potentials import pot_from_funct
from math import exp
from fem_base.exploit_fun import data_path

res_path=data_path
eV=1.60217663 *10**-19
'''for l in [50,100,200,500,1000]:
    for v in range(5):
        pot,q_pot=gauss_pot(eV,2.2,int(l/0.56),l*10**-9,0.15)
        name=res_path+'pre_interp_pot/pre_interp_potv'+str(v)+'l'+str(l)+'E1eVx15.npy'
        np.save(name,pot)
    np.save(res_path+'/Vq/Vql'+str(l),q_pot)
'''
v=6
for f in (lambda x,y:2*exp(-0.1*((x-8)**2+(y-8)**2))-3*exp(-0.3*((x-5.5)**2+(y-1)**2))+4/((x-2)**2+(y-5)**2+1),lambda x,y:4*(1-1/((x-5)**2/2+(y-4)**2/4+1))):
    v+=1
    feV= lambda x,y: eV*f(x,y)/1000
    for l in [50,100,200,500,1000]:
        N=int(l/0.56)
        pot=pot_from_funct(feV,N)
        name=os.path.realpath(os.path.join(data_path,'pre_interp_pot','pre_interp_potv'+str(v)+'l'+str(l)+'E1eVx15.npy'))
        np.save(name,pot)