import numpy as np
import sys
import time

from so_pair import trid, build_pot, get_mu, cn, get_sig, get_m, iyt

tol = 1e-11
cpar = [5e6,1e3]

#

Dx = (np.float(sys.argv[1]),np.float(sys.argv[3]))
Dy = (np.float(sys.argv[2]),np.float(sys.argv[3]))
Dt = np.float(sys.argv[4])
g = (np.float(sys.argv[5]),np.float(sys.argv[6]))

q = np.float(sys.argv[8])
r0 = (np.float(sys.argv[9]),np.float(sys.argv[10]))
ind = int(sys.argv[11])
vp = int(sys.argv[12])

V0 = 5e2
R0 = .9 * Dx[0]

N0 = np.float(sys.argv[7]) * np.pi * pow(R0,2) * pow(4.*Dx[0]*Dy[0],-1)

qc = q * N0 * pow(np.pi*pow(R0,2),-1)

dlist = [False, True]
vlist = ["tri", "pcv", "mhv", "mhv_r"]

#

ti = time.time()
x, y, psi_gnd, mu, mu_dat, pop_dat = iyt(Dx=Dx,Dy=Dy,idt=Dt,g=g,q=q,N=N0,R0=R0,V0=V0,rv=r0,cpar=cpar,tol=tol,d_flag=dlist[vp],v_flag=vlist[ind])
print('total run time: ' + repr(time.time() - ti))

#

np.savez('data/pair/gn_'+repr(int(g[0]))+'_gs_m'+repr(abs(int(g[1])))+'_N0_'+get_m(N0)+'_q_'+get_m(q)+'_r0_'+get_m(r0[0])+'_'+get_m(r0[1])+'_Dx_'+get_m(Dx[0])+'_'+vlist[ind]+'.npz',x=x,y=y,psi_gnd=psi_gnd,mu=mu,mu_dat=mu_dat,pop_dat=pop_dat,Dx=Dx,Dy=Dy,Dt=Dt,g=g,N0=N0,q=q,r0=r0,V0=V0,R0=R0,tol=tol,cpar=cpar,v_flag=vlist[ind])

#
