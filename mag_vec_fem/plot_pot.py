from exploit_fun import *
from matplotlib import pyplot as plt
res_path=res_path_f()

lnm=200
pot_version=2
nameV=res_path+'/pre_interp_pot/'+'pre_interp_potv'+str(pot_version)+'l'+str(lnm)+'E1eVx15.npy'
V_preinterp=np.load(nameV)

plt.close()
plt.imshow(V_preinterp, interpolation='bilinear')
plt.show()
