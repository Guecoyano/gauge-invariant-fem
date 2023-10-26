import numpy as np
from scipy.stats import linregress
import os
import sys
from fem_base.exploit_fun import *

params = [
    ("load_file", True, None),
    ("dir_to_save", True, None),
    ("name_save", True, None),
]
load_file=dir_to_save=name_save=None

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default

histograms=np.load(load_file,allow_pickle=True)["histograms_proba"]
energies=np.load(load_file,allow_pickle=True)["eig_val"]

slopes,rvalues,pvalues,stderrs=[],[],[],[]
for hist in histograms:
    upper_bound=hist[:,1]<10000
    lower_bound=100<hist[:,1]
    selection=upper_bound & lower_bound
    reduced_hist=hist[selection,:]
    slope, _, rvalue, pvalue, stderr=linregress(-np.log(reduced_hist[:,0]),np.sqrt(reduced_hist[:,1]))
    slopes.append(slope), rvalues.append(rvalue),pvalues.append(pvalue),stderrs.append(stderr)
np.savez_compressed(os.path.join(dir_to_save,name_save),slopes=slopes,rvalues=rvalues,pvalues=pvalues,stderrs=stderrs, eig_val=energies)