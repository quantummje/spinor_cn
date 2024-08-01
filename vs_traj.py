import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

#
def vort_sort(dat_in):
    #
    out = np.zeros((dat_in.shape[0],dat_in.shape[1],dat_in.shape[2]))
    #
    for x in range(0,dat_in.shape[1]):
        #
        this_vort = x
        out[0][this_vort,:] = dat_in[0][this_vort,:]
        #
        for t in range(1,dat_in.shape[0]):
            #
            r_tmp = []
            #
            for y in range(0,dat_in.shape[1]):
                #
                rn = np.sqrt((dat_in[t][y,0]-out[t-1][this_vort,0])**2 + (dat_in[t][y,1]-out[t-1][this_vort,1])**2)
                r_tmp.append(rn)
                #
            next_vort = r_tmp.index(min(r_tmp))
            #
            out[t][this_vort,:] = dat_in[t][next_vort,:]
            #
        #
    #
    return out
#

#
def track(psi,r0,x,y):
    #
    den = np.abs(psi)**2
    vort_pos = [[],[]]
    vort_ind = [[],[]]
    m_den = .01 * np.max(np.max(den))
    #
    for ii in range(0,psi.shape[0]):
        #
        for jj in range(0,psi.shape[1]):
            #
            if np.sqrt(x[ii]**2 + y[jj]**2) <= r0:
                #
                if (den[ii+1][jj] > den[ii][jj]) and (den[ii][jj] < den[ii-1][jj]) and (den[ii][jj+1] > den[ii][jj]) and (den[ii][jj] < den[ii][jj-1]) and 0. < den[ii,jj] < m_den:
                    #
                    vort_pos = np.append(vort_pos,[[x[ii]],[y[jj]]],axis=1)
                    vort_ind = np.append(vort_ind,[[int(ii)],[int(jj)]],axis=1)
                    #
                #
            #
        #
    #

    vort_ind = np.array(vort_ind).astype(int)

    if len(vort_pos[0]) !=0 and len(vort_pos[1]) !=0 and vort_pos.shape[1] > 2:
        #
        v_den = np.zeros((2,vort_pos.shape[1]))
        v_tmp = np.zeros((2,2))
        i_tmp = np.zeros((2,2))

        for jj in range(0,v_den.shape[1]):
            #
            v_den[0,jj] = den[vort_ind[0,jj],vort_ind[1,jj]]
            v_den[1,jj] = int(jj)
        #

        vs = sorted(v_den.T, key = lambda z:float(z[0]))

        if vort_pos[1,int(vs[1][1])] < vort_pos[1,int(vs[0][1])]: #flip the gtr/lst when changing spin comps.
            #
            v_tmp[0,0] = vort_pos[0,int(vs[1][1])] #y1
            v_tmp[1,0] = vort_pos[1,int(vs[1][1])] #y2

            v_tmp[0,1] = vort_pos[0,int(vs[0][1])] #x1
            v_tmp[1,1] = vort_pos[1,int(vs[0][1])] #x2

            i_tmp[0,0] = vort_ind[0,int(vs[1][1])]
            i_tmp[1,0] = vort_ind[1,int(vs[1][1])]

            i_tmp[0,1] = vort_ind[0,int(vs[0][1])]
            i_tmp[1,1] = vort_ind[1,int(vs[0][1])]

        #
        elif vort_pos[1,int(vs[1][1])] > vort_pos[1,int(vs[0][1])]:
            #
            v_tmp[0,0] = vort_pos[0,int(vs[0][1])] #y1
            v_tmp[1,0] = vort_pos[1,int(vs[0][1])] #y2

            v_tmp[0,1] = vort_pos[0,int(vs[1][1])] #x1
            v_tmp[1,1] = vort_pos[1,int(vs[1][1])] #x2

            i_tmp[0,0] = vort_ind[0,int(vs[0][1])]
            i_tmp[1,0] = vort_ind[1,int(vs[0][1])]

            i_tmp[0,1] = vort_ind[0,int(vs[1][1])]
            i_tmp[1,1] = vort_ind[1,int(vs[1][1])]

        #
        #print(repr(vort_pos.shape[1]))
        vort_pos = v_tmp
        vort_ind = i_tmp.astype(int)
    #

    return vort_pos, vort_ind
#

#
def get_vsep(dat_in):
    #
    dat_sep = np.zeros((1,dat_in.shape[0]))
    #
    for jj in range(0,dat_sep.shape[1]):
        #
        dat_sep[0,jj] = pow(pow(dat_in[jj][1,0]-dat_in[jj][1,1],2) + pow(dat_in[jj][0,0]-dat_in[jj][0,1,],2),0.5)
    #
    return dat_sep
#

#
def rmv_zs(dat_in):
    #
    a_00 = list(dat_in[:,0,0])
    a_01 = list(dat_in[:,0,1])
    a_10 = list(dat_in[:,1,0])
    a_11 = list(dat_in[:,1,1])
    #
    to_del_a = []
    to_del_b = []
    #
    for jj in range(0,dat_in.shape[0]):
        #
        if a_10[jj] == 0. and a_00[jj] == 0.:
            #
            to_del_a = np.append(to_del_a,int(jj))
        #
        if a_11[jj] == 0. and a_01[jj] == 0.:
            #
            to_del_b = np.append(to_del_b,int(jj))
        #
    #
    for jj in range(0,len(to_del_a)):
        #
        a_10.pop(int(to_del_a[jj]-jj))
        a_00.pop(int(to_del_a[jj]-jj))
    #
    for jj in range(0,len(to_del_b)):
        #
        a_11.pop(int(to_del_b[jj]-jj))
        a_01.pop(int(to_del_b[jj]-jj))
    #
    out = np.array([[a_00,a_10],[a_01,a_11]]).T
    #
    return out
#

dat = np.load('dynamics/gn_1000_gs_m10_N0_50p9_q_0p5_r0_4_0_Dx_11_pcv_T_20.npz',allow_pickle=True)

sptm = dat['sptm']
R0 = dat['R0']
x, y = dat['x'], dat['y']

tmp_tim = [[],[]]
sFlag = True
mf_ind = int(2) # spin index

#
for jj in range(0,sptm.shape[0]):
    #
    v_p, v_i = track(psi=sptm[jj,mf_ind],r0=R0,x=x,y=y)
    #
    if len(v_p[0]) != 0 and len(v_p[1]) != 0:
        #
        print('iteration ' + repr(jj) + ' vorts ' + repr(v_p.shape[1]))
        #print(v_p)
        #
        if v_p.shape[1] != 1 and sFlag:
            #
            tmp_tim = np.append(tmp_tim,v_p,axis=1)
            #print(v_p)
        #
        elif v_p.shape[1] == 1:
            #
            if sFlag == True:
                #
                end_jj = jj
                print('end_jj = ' + repr(jj))
            #
            sFlag = False
        #
    #
#

dat_tim = np.zeros((end_jj+5,2,2)) #tmp_tim.shape[1]/2 instead of end_jj
ts = np.linspace(0,np.float(dat_tim.shape[0])/sptm.shape[0]*dat['T'],dat_tim.shape[0])

for jj in range(0,tmp_tim.shape[1],2):
    #
    dat_tim[int(jj/2)][:,:] = tmp_tim[:,jj:jj+2]
#

vs = vort_sort(dat_in=dat_tim)

freeD = plt.figure(figsize=(4,7))
ax = freeD.gca(projection='3d')
#
ax.plot(dat_tim[:,1,0][:],dat_tim[:,0,0][:],ts[:],marker='.',color='tab:blue',label=r'$(x_{+},y_{+}) |\psi_{+1}|^2$')
ax.plot(dat_tim[:,1,1][:],dat_tim[:,0,1][:],ts[:],marker='.',color='tab:orange',label=r'$(x_{-},y_{-}) |\psi_{+1}|^2$')
#
#ax.plot(dat_tim[:,1,0][:259],dat_tim[:,0,0][:259],ts[:259],marker='.',color='tab:green',label=r'$(x_{-},y_{-}) |\psi_{-1}|^2$')
#ax.plot(dat_tim[:,1,1][:259],dat_tim[:,0,1][:259],ts[:259],marker='.',color='tab:red',label=r'$(x_{+},y_{+}) |\psi_{-1}|^2$')
#
ax.set_zlim([0,ts[-1]])
ax.set_xlabel(r'$x$',fontsize=14)
ax.set_ylabel(r'$y$',fontsize=14)
ax.set_zlabel(r'$t$',fontsize=14)
ax.set_xlim([-dat['Dx'][0],dat['Dx'][0]])
ax.set_ylim([-dat['Dy'][0],dat['Dy'][0]])
ax.view_init(elev=10., azim=45.)
ax.legend()
#plt.tight_layout()
freeD.show()

v_sep = get_vsep(dat_in=dat_tim)

sp1 = CubicSpline(ts, v_sep[0,:])

f1 = plt.figure(figsize=(4,3))
plt.plot(ts[::2],v_sep[0,::2],'.',color='black')
#plt.plot(ts[::4],v_sep_m[0,:][::4],'o',markerfacecolor='none')
plt.plot(ts[::2],sp1(ts[::2]))
#plt.xlim([0.,75.])
#plt.ylim([6.,16.])
f1.show()

f1, ax = plt.subplots(1,1,figsize=(4,3))
#
t_out = rmv_zs(dat_in=dat_tim)
#
ax.plot(t_out[:,1,0],t_out[:,0,0],'.',color='tab:blue',label=r'$(x_{+},y_{+}) |\psi_{+1}|^2$')
ax.plot(t_out[:,1,1],t_out[:,0,1],'o',color='tab:blue',label=r'$(x_{-},y_{-}) |\psi_{+1}|^2$',markerfacecolor='none',markersize=4)

#ax.plot(dat_tim[:,1,0][::2],dat_tim[:,0,0][::2],'.',color='tab:blue',label=r'$(x_{+},y_{+}) |\psi_{+1}|^2$')
#ax.plot(dat_tim[:,1,1][::2],dat_tim[:,0,1][::2],'o',color='tab:blue',label=r'$(x_{-},y_{-}) |\psi_{+1}|^2$',markerfacecolor='none',markersize=4)
#
#ax.plot(dat_tim[:,1,0][::4],dat_tim[:,0,0][::4],'.',color='tab:red',label=r'$(x_{-},y_{-}) |\psi_{-1}|^2$')
#ax.plot(dat_tim[:,1,1][::4],dat_tim[:,0,1][::4],'o',color='tab:red',label=r'$(x_{+},y_{+}) |\psi_{-1}|^2$',markerfacecolor='none')
#
ax.plot(R0*np.cos(np.linspace(0,2*np.pi,100)),R0*np.sin(np.linspace(0,2*np.pi,100)),'-',color='gray',linewidth=1)
ax.set_xlabel(r'$x$',fontsize=14)
ax.set_ylabel(r'$y$',fontsize=14)
ax.legend()#fontsize=9,loc="center right",ncol=2, bbox_to_anchor=(0.86,0.695))
ax.axis('equal')
#
#ax[1].plot(,'+',color='tab:blue')
#

plt.tight_layout()
f1.show()

#f1 = plt.figure(figsize=(5,4))
#plt.plot(ts_1[::4],v_sep_p1[0,:][::4],'.',color='black',label=r'${\rm PCV\ sep.} |\psi_{+1}|^2 N\sim 25$',markersize='8')
#plt.plot(ts_1,s1_p,label=r'cubic interpolation $(N\sim 25)$')
#plt.plot(ts_2[::8],v_sep_m2[0,:][::8],'+',color='black',label=r'${\rm PCV\ sep.} |\psi_{+1}|^2 N\sim 6$',markersize='8')
#plt.plot(ts_2,s2_m,label=r'cubic interpolation $(N\sim 4)$')
#plt.xlabel(r'$t/t_0$',fontsize=14)
#plt.ylabel(r'$x/\xi_0$',fontsize=14)
#plt.xlim([0.,1.05*np.max(ts_1)])
#plt.ylim([0.,1.05*np.max(v_sep_p1)])
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.legend(fontsize=11)
#plt.grid()
#plt.tight_layout()
#f1.show()

#f2 = plt.figure(figsize=(6,3))
#
#plt.plot(ts_q_0,v_sep_q_0,'.',color='black')
#plt.plot(ts_q_0,sp1_q_0,label=r'$q/q_0=0$')
#
#plt.plot(ts_q_0p125,v_sep_q_0p125,'.',color='black')
#plt.plot(ts_q_0p125,sp1_q_0p125,label=r'$q/q_0=0.125$')
#
#plt.plot(ts_q_0p25,v_sep_q_0p25,'.',color='black')
#plt.plot(ts_q_0p25,sp1_q_0p25,label=r'$q/q_0=0.25$')
#
#plt.plot(ts_q_0p5,v_sep_q_0p5,'.',color='black')
#plt.plot(ts_q_0p5,sp1_q_0p5,label=r'$q/q_0=0.5$')
#
#plt.plot(ts_q_0p75,v_sep_q_0p75,'.',color='black')
#plt.plot(ts_q_0p75,sp1_q_0p75,label=r'$q/q_0=0.75$')
#
#plt.plot(ts_q_1p0,v_sep_q_1p0,'.',color='black')
#plt.plot(ts_q_1p0,sp1_q_1p0,label=r'$q/q_0=1.0$')
#
#plt.xlabel(r'$t$',fontsize=14)
#plt.ylabel(r'$\Delta r=r_{+}-r_{-}$',fontsize=14)
#plt.tick_params(axis='both', which='major', labelsize=14)
#plt.xlim([0.,np.max(ts_q_0)])
#plt.ylim([0.,8.25])
#
#plt.legend(frameon=False,ncol=2,labelspacing=0,columnspacing=0.5)
#plt.tight_layout()
#f2.show()
