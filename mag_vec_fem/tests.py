from exploit_fun import *
import tracemalloc
tracemalloc.start()
res_path=res_path_f()


lnm=100
h=0.002
pot_version=1
gauge='Sym'
l=lnm*10**-9


VmeV,Th=vth_data(res_path,lnm,h,pot_version)
n=0
snap=[(tracemalloc.take_snapshot()).statistics('lineno')[:10]]
top_stats=[]
for V_maxmeV in (5000,):
    N_eig=100
    for B in (12,):
        getsave_eig(N_eig,lnm,B,VmeV,V_maxmeV,Th,h,pot_version,gauge,res_path)
        snap+=[(tracemalloc.take_snapshot()).statistics('lineno')[:10]]
        #top_stats += [snap[n+1].compare_to(snap[n], 'lineno')]
        n+=1
        for k in snap[-1]:
            print(k)

print("[ Top 10 differences ]")
for k in range(n):
    print('k=',k)
    for stat in snap[k][:6]:
        print(stat)
