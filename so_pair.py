import numpy as np
import time

#
def trid(A,Au,Al,d,NN):
    #
    gamma = [0. + 0j] #gamma = np.zeros(int(N),dtype=float)

    bta = A[0]
    tmp = [d[0] / bta]

    Au = np.concatenate((Au,[0. + 0j]))
    Al = np.concatenate(([0. + 0j],Al))

    for jj in range(1,NN,1):
        #
        gamma.append( Au[jj-1] / bta )
        bta = A[jj] - Al[jj]*gamma[jj]

        tmp.append( (d[jj] - Al[jj]*tmp[jj-1]) / bta )
    #

    for jj in range(2,NN+2,1):
        kk = NN - jj
        tmp[kk] = tmp[kk] - gamma[kk+1]*tmp[kk+1]
    #

    return np.array(tmp)
#

#
def build_pot(Dx,Dy,R0):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    xx, yy = np.meshgrid(x,y)
    #
    r = pow(pow(xx,2) + pow(yy,2),.5)
    #
    hvs_pr = 0.5*(1.0 + np.tanh((r + R0)/0.25))
    hvs_mr = 0.5*(1.0 + np.tanh((r - R0)/0.25))
    #
    pot = 1.0 - (hvs_pr - hvs_mr)
    #
    return pot, x, y
#

#
def get_mu(psi_in,Dx,Dy,q,g,R0,V0):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #

    pot, _, _, = build_pot(Dx=Dx,Dy=Dy,R0=R0)

    V = V0 * pot

    #

    ke = np.zeros(3)

    ei_p = np.zeros(3)
    ei_v = np.zeros(3)
    ei_s = np.zeros(3)
    ei_t = np.zeros(3)

    Na = np.zeros(3)
    mu = np.zeros(3)
    #
    for jj in range(1,Nx-1):
        #
        for kk in range(1,Ny-1):
            #
            dn = np.abs(psi_in[0][jj,kk])**2 + np.abs(psi_in[1][jj,kk])**2 + np.abs(psi_in[2][jj,kk])**2
            #
            for l in range(0,3):
                #
                kx = np.abs((psi_in[l][jj+1,kk] - psi_in[l][jj-1,kk])/(2*dx))**2
                ky = np.abs((psi_in[l][jj,kk+1] - psi_in[l][jj,kk-1])/(2*dy))**2
                #
                ke[l] = ke[l] + dx*dy*(kx + ky)
                #
                ei_p[l] = ei_p[l] + dx*dy*V[kk,jj]*np.abs(psi_in[l][jj,kk])**2
                #
                ei_v[l] = ei_v[l] + g[0]*dx*dy*dn*np.abs(psi_in[l][jj,kk])**2
                #
            #
        #
    #
    ei_s[0] = g[1]*dx*dy*np.trapz(np.trapz((np.abs(psi_in[0])**2 + np.abs(psi_in[1])**2 - np.abs(psi_in[2])**2)*np.abs(psi_in[0])**2))
    ei_s[1] = g[1]*dx*dy*np.trapz(np.trapz((np.abs(psi_in[0])**2 + np.abs(psi_in[2])**2)*np.abs(psi_in[1])**2))
    ei_s[2] = g[1]*dx*dy*np.trapz(np.trapz((np.abs(psi_in[2])**2 + np.abs(psi_in[1])**2 - np.abs(psi_in[0])**2)*np.abs(psi_in[2])**2))

    ei_t[0] = np.real(g[1]*dx*dy*np.trapz(np.trapz(np.conj(psi_in[0])*np.conj(psi_in[2])*pow(psi_in[1],2))))
    ei_t[1] = np.real(2*g[1]*dx*dy*np.trapz(np.trapz(pow(np.conj(psi_in[1]),2)*psi_in[0]*psi_in[2])))
    ei_t[2] = np.real(g[1]*dx*dy*np.trapz(np.trapz(np.conj(psi_in[2])*np.conj(psi_in[0])*pow(psi_in[1],2))))
    #

    #
    Na[0] = dx*dy*np.trapz(np.trapz(np.abs(psi_in[0])**2))
    Na[1] = dx*dy*np.trapz(np.trapz(np.abs(psi_in[1])**2))
    Na[2] = dx*dy*np.trapz(np.trapz(np.abs(psi_in[2])**2))
    #
    for jj in range(0,3):
        #
        if Na[jj] == 0:
            #
            mu[jj] = 0.
        #
        else:
            #
            mu[jj] = (-0.5*ke[jj] + np.abs(jj-1)*q*Na[jj] + ei_p[jj] + ei_v[jj] + ei_s[jj] + ei_t[jj])/Na[jj]
            #mu[1] = (-k0*ke[1] + ei_p[1] + ei_v[1] + ei_s[1] + ei_t[1])/Na[1]
            #mu[2] = (-k0*ke[2] + q*Na[2] + ei_p[2] + ei_v[2] + ei_s[2] + ei_t[2])/Na[2]
        #
    #
    return mu
    #
#

#
def cn(p_in,u_x,d_xp,d_xm,l_x,u_y,d_yp,d_ym,l_y,Nx,Ny):
    #

    p0 = p_in.reshape(Nx*Ny,order='C') #y-dir.

    p1 = np.concatenate(([0.],l_y*p0[:-1])) + np.concatenate((u_y*p0[1:],[0.])) + d_ym*p0

    p2 = p1.reshape(Ny,Nx,order='C').reshape(Nx*Ny,order='F') #x-dir.

    p3 = np.concatenate(([0.],l_x*p2[:-1])) + np.concatenate((u_x*p2[1:],[0.])) + d_xm*p2

    p4 = trid(A=d_xp,Au=-u_x,Al=-l_x,d=p3,NN=Nx*Ny)

    p5 = p4.reshape(Ny,Nx,order='F').reshape(Nx*Ny,order='C')

    p6 = trid(A=d_yp,Au=-u_y,Al=-l_y,d=p5,NN=Nx*Ny)

    #
    return p6
#

#
def get_sig(n_p,n_z,n_m,N):
    #
    sig = np.zeros(3)
    M = n_p - n_m

    sig[1] = pow((N**2 - M**2)*pow(N*n_z + pow(4.*(N**2 - M**2)*n_p*n_m + (M*n_z)**2,.5),-1) ,.5) #\sgima_0
    sig[0] = pow((N + M - n_z*sig[1]**2)/(2.0*n_p),.5) #\sigma_+1
    sig[2] = pow((N - M - n_z*sig[1]**2)/(2.0*n_m),.5) #\sigma_-1

    #
    return sig
#

#
def get_m(num_in):
    #
    if num_in % 1 != 0:
        #
	if num_in > 0.:
            #
            out = repr(int(num_in)) + 'p' + repr(int(round(1e1*(num_in % 1))))
        #
	elif num_in < 0.:
            #
            out = 'm' + repr(int(abs(num_in))) + 'p' + repr(int(round(1e1*(abs(num_in) % 1))))
        #
    #
    else:
	#
	out = repr(int(num_in))
    #
    return out
#

#
def iyt(Dx,Dy,idt,g,q,N,R0,V0,rv,cpar,tol,d_flag,v_flag):
    #
    Nx = int(1 + 2*Dx[0]/Dx[1])
    Ny = int(1 + 2*Dy[0]/Dy[1])

    NN = Nx * Ny
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    xx, yy = np.meshgrid(x, y)

    rr_1 = pow((xx-rv[0])**2 + (yy-rv[1])**2,.5)
    rr_2 = pow((xx+rv[0])**2 + (yy-rv[1])**2,.5)

    DD = Dx[1] * Dy[1]

    if V0 != 0:
        #
        pot, _, _, = build_pot(Dx=Dx,Dy=Dy,R0=R0)
        vort_0 = pow(np.tanh(2*rr_1),2) * pow(np.tanh(2*rr_2),2) * (1. - pot)
        disc_0 = 1. - pot
        V = V0 * pot.reshape(NN,order='F')
    #
    else:
        #
        psi_0 = pow(np.tanh(2*rr),2) * np.ones((Nx,Ny))
        V = 0.
    #

    xl = idt*pow(4.0*pow(Dx[1],2),-1) * np.kron(np.ones(Nx),np.concatenate((np.ones(int(Ny-2)),[1.],[0.])))[:-1]
    xu = idt*pow(4.0*pow(Dx[1],2),-1) * np.kron(np.ones(Nx),np.concatenate(([1.],np.ones(int(Ny-2)),[0.])))[:-1]

    yl = idt*pow(4.0*pow(Dy[1],2),-1) * np.kron(np.ones(Ny),np.concatenate((np.ones(int(Nx-2)),[1.],[0.])))[:-1]
    yu = idt*pow(4.0*pow(Dy[1],2),-1) * np.kron(np.ones(Ny),np.concatenate(([1.],np.ones(int(Nx-2)),[0.])))[:-1]

    A_0x = 0.5*idt*(pow(Dx[1],-2) + q + V)

    A_0x_p = np.ones(NN) + A_0x
    A_0x_m = np.ones(NN) - A_0x

    A_0z = 0.5*idt*(pow(Dx[1],-2) + V)

    A_0z_p = np.ones(NN) + A_0z
    A_0z_m = np.ones(NN) - A_0z

    A_0y = 0.5*idt*pow(Dy[1],-2)

    A_0y_p = np.ones(NN) + A_0y
    A_0y_m = np.ones(NN) - A_0y

    if d_flag == True:
	#
	vp = np.arctan2(yy-rv[1],xx-rv[0]) - np.arctan2(yy-rv[1],xx+rv[0])
    #
    elif d_flag == False:
	#
	vp = np.arctan2(yy-rv[1],xx-rv[0]) + np.arctan2(yy-rv[1],xx+rv[0])
    #

    p0_p = disc_0 * (1. + 1e-6*np.random.rand(Nx,Ny))
    p0_z = disc_0 * (1. + 1e-6*np.random.rand(Nx,Ny))
    p0_m = disc_0 * (1. + 1e-6*np.random.rand(Nx,Ny))

    #

    if v_flag == 'tri':
        #
        phi_p = np.exp(1j*vp)
        phi_z = np.exp(1j*vp)
        phi_m = np.exp(1j*vp)
    #
    elif v_flag == "pcv":
        #
        phi_p = np.exp(-1j*vp)
        phi_z = np.ones((Nx,Ny))
        phi_m = np.exp(1j*vp)
    #
    elif v_flag == "mhv":
        #
        phi_p = np.ones((Nx,Ny))
        phi_z = np.exp(1j*vp)
        phi_m = np.exp(2j*vp)
    #
    elif v_flag == "mhv_r":
	#
	phi_p = np.exp(-2j*np.arctan2(yy-rv[1],xx-rv[0]))
	phi_z = np.exp(1j*vp)
	phi_m = np.exp(2j*np.arctan2(yy-rv[1],xx+rv[0]))
    #

    n_p = np.real(DD*np.trapz(np.trapz(np.conj(p0_p)*p0_p)))
    n_z = np.real(DD*np.trapz(np.trapz(np.conj(p0_z)*p0_z)))
    n_m = np.real(DD*np.trapz(np.trapz(np.conj(p0_m)*p0_m)))

    sig = get_sig(n_p=n_p,n_z=n_z,n_m=n_m,N=N)

    p_p = sig[0] * p0_p
    p_z = sig[1] * p0_z
    p_m = sig[2] * p0_m

    p_p = np.abs(p_p) * phi_p
    p_z = np.abs(p_z) * phi_z
    p_m = np.abs(p_m) * phi_m

    n_p = np.real(DD*np.trapz(np.trapz(np.conj(p_p)*p_p)))
    n_z = np.real(DD*np.trapz(np.trapz(np.conj(p_z)*p_z)))
    n_m = np.real(DD*np.trapz(np.trapz(np.conj(p_m)*p_m)))

    print('initial norm \psi_+1: ' + repr(n_p))
    print('initial norm \psi_0: ' + repr(n_z))
    print('initial norm \psi_-1: ' + repr(n_m))

    print('initial total norm: ' + repr(n_p+n_z+n_m))
    print('initial magnetization: ' + repr(n_p-n_m))

    mu = get_mu(psi_in=np.concatenate((p_p,p_z,p_m)).reshape(3,Nx,Ny),Dx=Dx,Dy=Dy,q=q,g=g,R0=R0,V0=V0)

    print('chem. pot. \psi_+1: ' + repr(mu[0]))
    print('chem. pot. \psi_0: ' + repr(mu[1]))
    print('chem. pot. \psi_-1: ' + repr(mu[2]))
    print('-------------------------------------')

    mu = np.random.rand(3)
    mu_old = np.random.rand(3)
    mu_err = np.random.rand(3)

    t0 = time.time()

    #

    ii = 0
    iimax = cpar[0]

    mu_dat = [[],[],[]]
    pop_dat = [[],[],[]]

    iFlag = True

    #

    while iFlag == True:
        #

        # spin-1 single-particle Crank-Nicolson steps

        cn_p = cn(p_in=p_p,u_x=xu,d_xp=A_0x_p,d_xm=A_0x_m,l_x=xl,u_y=yu,d_yp=A_0y_p,d_ym=A_0y_m,l_y=yl,Nx=Nx,Ny=Ny)

        cn_z = cn(p_in=p_z,u_x=xu,d_xp=A_0z_p,d_xm=A_0z_m,l_x=xl,u_y=yu,d_yp=A_0y_p,d_ym=A_0y_m,l_y=yl,Nx=Nx,Ny=Ny)

        cn_m = cn(p_in=p_m,u_x=xu,d_xp=A_0x_p,d_xm=A_0x_m,l_x=xl,u_y=yu,d_yp=A_0y_p,d_ym=A_0y_m,l_y=yl,Nx=Nx,Ny=Ny)

        # mass den :

        d_p = np.abs(cn_p)**2
        d_z = np.abs(cn_z)**2
        d_m = np.abs(cn_m)**2

        dn = d_p + d_z + d_m

        p_p = np.exp(-idt*(g[0]*dn + g[1]*(d_p + d_z -d_m))) * cn_p

        p_z = np.exp(-idt*(g[0]*dn + g[1]*(d_p + d_m))) * cn_z

        p_m = np.exp(-idt*(g[0]*dn + g[1]*(-d_p + d_z + d_m))) * cn_m

        # ints: spin-ex.

	d_p = np.conj(p_p) * p_p
        d_z = np.conj(p_z) * p_z
        d_m = np.conj(p_m) * p_m

        Om = g[1]*idt*np.abs(p_z)*pow(np.conj(p_p)*p_p + np.conj(p_m)*p_m,.5) + 1e-40	
        sinh_f = -idt*g[1]*np.sinh(Om)*pow(Om,-1)
        cosh_f = pow(idt*g[1],2)*(np.cosh(Om)-1.0)*pow(Om,-2)*d_z

        p_pe = p_p + sinh_f*p_z*np.conj(p_m)*p_z + 2.*cosh_f*p_p*d_m

        p_ze = 2.*sinh_f*np.conj(p_z)*p_m*p_p + (1. + cosh_f*(d_p + d_m))*p_z

        p_me = 2.*cosh_f*d_p*p_m + sinh_f*p_z*np.conj(p_p)*p_z + p_m

        # matrix form

        p_p = p_pe.reshape(Ny,Nx,order='C')

        p_z = p_ze.reshape(Ny,Nx,order='C')

        p_m = p_me.reshape(Ny,Nx,order='C')

        #

        n_p = np.real(DD*np.trapz(np.trapz(np.conj(p_p)*p_p)))
        n_z = np.real(DD*np.trapz(np.trapz(np.conj(p_z)*p_z)))
        n_m = np.real(DD*np.trapz(np.trapz(np.conj(p_m)*p_m)))

        sig = get_sig(n_p=n_p,n_z=n_z,n_m=n_m,N=N)

        p_p = sig[0] * p_p
        p_z = sig[1] * p_z
        p_m = sig[2] * p_m

        p_p = np.abs(p_p) * phi_p
        p_z = np.abs(p_z) * phi_z
        p_m = np.abs(p_m) * phi_m

        #
        if divmod(ii,cpar[1])[1] == 0 and ii != 0:
            #
            psi_it = np.concatenate((p_p,p_z,p_m)).reshape(3,Nx,Ny)

            mu = get_mu(psi_in=psi_it,Dx=Dx,Dy=Dy,q=q,g=g,R0=R0,V0=V0)
            #
            mu_err = np.abs((mu - mu_old)/mu)
            mu_dat = np.append(mu_dat,[[mu_err[0]],[mu_err[1]],[mu_err[2]]],axis=1)
            #
            pop_dat = np.append(pop_dat,[[n_p],[n_z],[n_m]],axis=1)

            if mu_err[0] < tol and mu_err[1] < tol and mu_err[2] < tol:
                #
                iFlag = False
            #
            elif ii >= iimax:
                #
                iFlag = False
            #

            print('chem. pot. \psi_+1: ' + repr(mu[0]) + ' error: ' + repr(mu_err[0]))
            print('chem. pot. \psi_0: ' + repr(mu[1]) + ' error: ' + repr(mu_err[1]))
            print('chem. pot. \psi_-1: ' + repr(mu[2]) + ' error: ' + repr(mu_err[2]))

            n_p = DD*np.trapz(np.trapz(np.conj(p_p)*p_p))
            n_z = DD*np.trapz(np.trapz(np.conj(p_z)*p_z))
            n_m = DD*np.trapz(np.trapz(np.conj(p_m)*p_m))

            print('norm. \psi_+1: ' + repr(n_p))
            print('norm. \psi_0: ' + repr(n_z))
            print('norm. \psi_-1: ' + repr(n_m))

            print('total mass den.: ' + repr(n_p+n_z+n_m))
            print('magnetization: ' + repr(n_p-n_m))
            print('-------------------------------------')
            #
        #

        mu_old = mu
        ii += 1

        #
    #

    print('iterations: ' + repr(ii))
    psi_out = np.concatenate((p_p,p_z,p_m)).reshape(3,Ny,Nx)

    #

    return x, y, psi_out, mu, mu_dat, pop_dat
#
