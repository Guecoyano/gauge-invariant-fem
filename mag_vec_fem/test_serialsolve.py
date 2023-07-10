N_eigfinal=100
N_batch=10
solve=lambda target : eigsh((A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N_eig, sigma=target, which="LM")

n_computed_vectors=0
x_batch,w_batch=solve