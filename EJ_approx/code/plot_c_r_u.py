#========================================================
#Name: c_r_u.py
#Function: this code is designed for calculate correlation
# and response functions numerically for spin glass models.
# In this code, we implemente the solving of an integro-
# differential eqn. with initial conditions.
#======================================================== 
#Author: Gang Huang
#Data: 2020-5-27
#Version: 3
#==================

import numpy as np
import matplotlib.pyplot as plt

# 3-p shphere model
def f(q):
    """For 3-p sphere model."""
    return 0.5*q**3
def df2(q):
    """For 3-p sphere model."""
    return 3*q
def df1(q):
    """For 3-p sphere model."""
    return 1.5*q**2

# # fixed 3-4 model
# def f(q):
#     """For 3-4-p sphere model."""
#     return 0.5*(q**3 + q**4)
# def df2(q):
#     """For 3-4-p sphere model."""
#     return 3*q + 6*q**2
# def df1(q):
#     """For 3-4-p sphere model."""
#     return 1.5*q**2 + 2*q**3

def plot_corr_log(beta0,beta,h,t1=1):
    c = np.load('corr_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta))
    print("C=",c)
    print("C.T=",c.T)
    c_trans = c.T
    length = c_trans.shape[0]
    t = np.array([x for x in range(length-t1)])
    corr = c_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, corr, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:5.3f}; $\beta ={:5.3f}$; $t'={:6.3f}$".format(beta0,beta,t1*h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.00001, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_betastart{:5.3f}_beta{:5.3f}_t1{:5.3f}.png".format(beta0,beta,t1*h),bbox_inches = 'tight')

def plot_corr_log2(beta0,beta,step,h,t1):
    c = np.load('corr_beta0_{:5.3f}_beta{:5.3f}_step{:d}_h{:5.3f}.npy'.format(beta0,beta,step,h))
    print("C=",c)
    print("C.T=",c.T)
    c_trans = c.T
    length = c_trans.shape[0]
    t = np.array([x for x in range(length-t1)])
    corr = c_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*0.1, corr, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:5.3f}; $\beta ={:5.3f}$; $t'={:6.3f}$".format(beta0,beta,t1))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.00001, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_betastart{:5.3f}_beta{:5.3f}_step{:d}_h{:5.3f}_t1_{:5.3f}.png".format(beta0,beta,step,h,t1),bbox_inches = 'tight')
def plot_corr_log(beta0,beta,h,t1=1):
    c = np.load('corr_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta))
    print("C=",c)
    print("C.T=",c.T)
    c_trans = c.T
    length = c_trans.shape[0]
    t = np.array([x for x in range(length-t1)])
    corr = c_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, corr, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:5.3f}; $\beta ={:5.3f}$; $t'={:6.3f}$".format(beta0,beta,t1*h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.00001, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_betastart{:5.3f}_beta{:5.3f}_t1{:5.3f}.png".format(beta0,beta,t1*h),bbox_inches = 'tight')

def plot_resp_loglog(beta0,beta,h,t1=1):
    """t1 denotes t' """
    r = np.load('resp_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta))
    print("r=",r)
    print("r.T=",r.T)
    r_trans = r.T
    length = r_trans.shape[0]
    t = np.array([x for x in range(length-t1)])
    resp = r_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, resp, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:5.3f}; $\beta ={:5.3f}$; $t'={:5.3f}$".format(beta0,beta,t1*h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.0001, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"Response$(t,t')$")
    plt.savefig("resp_beta0_{:5.3f}_beta{:5.3f}_t1_{:5.3f}.png".format(beta0,beta,t1*h),bbox_inches = 'tight')

def plot_ener_log(beta0,beta,h):
    ener = np.load('ener_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta))
    print("Ener=",ener)
    length = ener.shape[0]
    t = np.array([x for x in range(length)])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, ener, "-", color="black",)
    length = ener.shape[0]
    print("Ener:", ener)
    x = h*np.arange(length)
    fig = plt.figure()
    plt.xscale('log')
    ax = fig.add_subplot(111)
    ax.plot(x, ener, "-", color="black",)
    ax.set_title(r"$N=\infty$; $\beta_0$={:5.3f}; $\beta ={:5.3f}$".format(beta0,beta))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(-1.00, 0.00)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E(t)$")
    plt.savefig("ener_beta0_{:5.3f}_beta{:5.3f}.png".format(beta0,beta),bbox_inches = 'tight')
    
def print_time(start_time):
    import datetime
    end = datetime.datetime.now()
    print("Start:",start_time)
    print("End",end)
    print("Total running time:",end-start_time)

if __name__ == "__main__":
    t1_list = [20,300,2048]
    beta0 = 0.000
    beta = 2.000
    step = 12
    h = 0.04
    for t1 in t1_list:
        plot_corr_log2(beta0,beta,step,h,t1) 
'''
if __name__ == "__main__":
    """Time start"""
    import datetime
    start = datetime.datetime.now()

    """Parameters and initial conditions."""
    beta = 1.5 #final Temperature
    Tf = 1/beta
    beta0 = 0.000
    h = 0.1
    h_squard = h ** 2
    step_exp = 7 
    n_tot = 2**step_exp
    t1_list = [2**i for i in range(0,step_exp)]
    #t1_list = np.linspace(0,2**step_exp*h,2**step_exp+1)
    n1_tot = n_tot

    u = np.zeros(n_tot)
    c = np.zeros((n_tot,n1_tot))
    r = np.zeros((n_tot,n1_tot))
    c_trans = c.T
    #If we want to calculate energy, we have to define the array 'ener'
    ener = np.zeros(n_tot)

    # Initial conditions
    for i in range(n_tot):  
        r[i][i] = 1
        c[i][i] = 1
    #ener[0] = - df1(c[0][0]) * r[0][0] - beta0 * f(c[0][0]) 
    ener[0] = - beta0 * f(c[0][0]) 
    #u[0] = Tf + df2(c[0][0])*r[0][0]*c[0][0] + df1(c[0][0])*r[0][0] + beta0 * df1(c[0][0]) * c[0][0]
    u[0] = Tf + beta0 * df1(c[0][0]) * c[0][0]
 
    #Start to calcualte C and R, iteratively.  
    for n in range(1,n_tot):
        #Define an auxinary array vX_nm1s, where the 'nm1s' means indics of X[n-1][s] 
        vc_nm1s = []  
        for s in range(n):
            vc_nm1s.append(df2(c[n-1][s]) * r[n-1][s])
        # transform list to np array    
        vc_nm1s = np.array(vc_nm1s) 
        
        for n1 in range(n):
            dc2 = np.zeros(n+1)
            #dc2_val = 0
            dc = np.zeros(n)
            #dr = np.zeros(n+1)
            dr_val = 0 
            du = np.zeros(n+1)
            #If want to calculate energy, the 'de' array is required [to accelerate the calculation].
            de = np.zeros(n+1)

            #For calculating faster, define the following constants 
            #du[0] = df2(c[0][n-1])*r[0][n-1] + df1(c[0][n-1])*r[0][n-1]
            du[0] = 0
            #dc[0] = h * h*df2(c[0][n-1]) * r[0][n-1] * c[n1][0]
            #dc2[0] = df1(c[0][0]) * r[0][0]
            #dc2[0] = 0
            #dr[0] = df2(c[n1][n-1]) * r[n1][n-1]* r[n1][n1]
            #dr[0] = 0
            
            #de[0] = - df1(c[0][n-1]) * r[0][n-1]  # If we calculate energy, we need this constant
            de[0] = 0
            
            ## calculate c(t) and r(t)
            #for s in range(0,n1+1):
            #    dc2_val += df1(c[n-1][s]) * r[n1][s]
            dc2[0] = df1(c[n-1][0]) * r[n1][0]
            for s in range(1,n1+1): 
                dc2[s] = dc2[s-1] + df1(c[n-1][s]) * r[n1][s]
            #integration for obtaining r(t),c(t): s in range(n):
            c[n][n1] = h_squard * np.sum(vc_nm1s * c[:,n1][:n]) + h_squard * dc2[n1] + c[n-1][n1] - h * u[n-1]*c[n-1][n1] + h * beta0 * df1(c[n-1][0])*c[n1][0]              
            #c[n][n1] = h_squard * np.sum( vc_nm1s * c[:,n1][:n]) + h_squard * dc2_val + c[n-1][n1] - h * u[n-1]*c[n-1][n1] + h * beta0 * df1(c[n-1][0])*c[n1][0]              
            #dr[n1] = 0 # assign dr[n1]
            #for s in range(n1+1,n):
            #    dr[s] = dr[s-1] + df2(c[n-1][s]) * r[n-1][s] * r[s][n1]
            for s in range(n1,n+1):
                dr_val += df2(c[n-1][s]) * r[n-1][s] * r[s][n1]    
            r[n][n1] = h_squard * dr_val + r[n-1][n1] - h * u[n-1]*r[n-1][n1]
            
            #=============================================================
            #After we know c[n][0] and r[n][0], we can calculate e[n],u[n]. 
            #=============================================================
            #Define an auxinary array vX_nm1s, where the 'nm1s' means indics of X[n-1][s] 
            #comp_vu_nm1s = [df2(c[n][0])*r[n][0] * c[n][0]  + df1(c[n][0])*r[n][0]]
            comp_vu_nm1s = []
            #comp_ve_nm1s = [- df1(c[n][0]) * r[n][0]]
            comp_ve_nm1s = []
            for s in range(0,n+1):
                comp_vu_nm1s.append(df2(c[n][s])*r[n][s] * c[n][s]  + df1(c[n][s])*r[n][s])
                comp_ve_nm1s.append( - df1(c[n][s]) * r[n][s])
            # transform list to np array    
            comp_vu_nm1s = np.array(comp_vu_nm1s) 
            comp_ve_nm1s = np.array(comp_ve_nm1s) 

            # Start to calculating u[n] and e[n]
            # Third : calculate u(t)
            #integration for obtain u(t): s in range(n):
            #for s in range(1,n):
            #    du[s] = du[s-1] + (df2(c[s][n-1])*r[s][n-1] + df1(c[s][n-1])*r[s][n-1])
            u[n] = h * np.sum(comp_vu_nm1s) + Tf + beta0 * df1(c[n][0])*c[n][0]

            # Finally: if we want to calculate energy
            #for s in range(1,n):
      
            #    de[s] = de[s-1] - df1(c[s][n-1]) * r[s][n-1]
            ener[n] = h * (np.sum(comp_ve_nm1s)) - beta0 * f(c[n][0])
    #write Energy 
    print("Energy:",ener)
    filename_ener = ''.join(['ener_beta0_',str(beta0),'_beta',str(beta),'.dat']) 
    with open(filename_ener, 'w') as f: # able to append data to file
        for s in range(n_tot):
            f.write(" ".join([str(s),str(ener[s]),'\n'])) 
        f.close() # You can add this but it is not mandatory
    #write C(t) 
    print("Correlation:",c)
    filename_c = ''.join(['corr_beta0_',str(beta0),'_beta',str(beta),'.dat']) 
    with open(filename_c, 'w') as f: # able to append data to file
        for s in range(n_tot):
            for t in range(n1_tot):
                f.write(" ".join([str(s),str(t),str(c[s][t]),'\n'])) 
            f.write("\n")
            f.write("\n")
        f.close() # You can add this but it is not mandatory
    #write R(t) 
    print("Response:",r)
    filename_r = ''.join(['resp_beta0_',str(beta0),'_beta',str(beta),'.dat']) 
    with open(filename_r, 'w') as f: # able to append data to file
        for s in range(n_tot):
            for t in range(n1_tot):
                f.write(" ".join([str(s),str(t),str(r[s][t]),'\n'])) 
            f.write("\n")
            f.write("\n")
        f.close() # You can add this but it is not mandatory
        
    #save all arrays
    np.save('ener_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta),ener)  
    np.save('corr_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta),c)
    np.save('resp_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta),r)
    np.save('u_beta0_{:5.3f}_beta{:5.3f}.npy'.format(beta0,beta),u)

    # # plot the correlation function
    #for t1 in t1_list:
    t1 = 0
    plot_corr_log(beta0,beta,h,t1) 

    # # plot the response function
    #for t1 in t1_list:
    plot_resp_loglog(beta0,beta,h,t1)

    # # plot ener
    plot_ener_log(beta0,beta,h)

    # #show the total time used
    end = datetime.datetime.now()
    print("Start:",start)
    print("End",end)
    print("Total running time:",end-start)
    '''
