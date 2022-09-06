import numpy as np
from fem_base.potentials import gauss_pot

eV=1.60217663 *10**-19
for l in [50,100,200,500,1000]:
    for v in range(5):
        pot,q_pot=gauss_pot(eV,2.2,int(l/0.56),l*10**-9,0.15)
        name='pre_interp_potv'+str(v)+'l'+str(l)+'E1eVx15.npy'
        np.save(name,pot)
    np.save('Vql'+str(l),q_pot)
