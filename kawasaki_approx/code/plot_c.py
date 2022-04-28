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

def plot_corr_log(beta,omega,step,h):
    c = np.load('corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step,h))
    print("C=",c)
    length = c.shape[0]
    t = np.array([x for x in range(length)])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, corr, "-", color="black",)
    ax.set_title(r"$\beta$={:4.2f}; $\omega ={:4.2f}$;$h={:4.2f}$".format(beta,omega,h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.xlim(0.01, 1000)
    plt.ylim(0.0, 1.01)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_K_beta{:4.2f}_omega{:4.2f}_step{:4.2f}_h{:4.2f}.pdf".format(beta,omega,step,h),bbox_inches = 'tight')

def plot_corr_log2(beta0,beta,step,h,t1):
    c = np.load('corr_beta0_{:4.2f}_beta{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta0,beta,step,h))
    print("C=",c)
    print("C.T=",c.T)
    c_trans = c.T
    length = c_trans.shape[0]
    t = np.array([x for x in range(length-t1)])
    corr = c_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(0.01, 1000)
    ax = fig.add_subplot(111)
    ax.plot(t*0.1, corr, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:4.2f}; $\beta ={:4.2f}$; $t'={:6.3f}$".format(beta0,beta,t1))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.0, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_betastart{:4.2f}_beta{:4.2f}_step{:d}_h{:4.2f}_t1_{:4.2f}.pdf".format(beta0,beta,step,h,t1),bbox_inches = 'tight')
def plot_corr_log(beta,omega,step,h):
    c = np.load('corr_beta0_{:4.2f}_beta{:4.2f}.npy'.format(beta0,beta))
    print("C=",c)
    print("C.T=",c.T)
    c_trans = c.T
    length = c_trans.shape[0]
    t = np.array([x for x in range(length)])
    corr = c_trans[t1][t1:]
    fig = plt.figure()
    plt.xscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, corr, "-", color="black",)
    ax.set_title(r"N=$\infty$; $\beta_0$={:4.2f}; $\beta ={:4.2f}$; $t'={:6.3f}$".format(beta0,beta,t1*h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.xlim(0.01, 1000)
    plt.ylim(0.00, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t,t')$")
    plt.savefig("corr_betastart{:4.2f}_beta{:4.2f}_t1{:4.2f}.pdf".format(beta0,beta,t1*h),bbox_inches = 'tight')

def plot_c_log(beta,omega,step,h):
    c = np.load('corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step,h))
    length = c.shape[0]
    t = np.array([x for x in range(length)])
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, c, "-", color="black",)
    length = c.shape[0]
    print("Corr:", c)
    x = h*np.arange(length)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, c, "-", color="black",)
    ax.set_title(r"$\beta$={:4.2f}; $\omega ={:4.2f}$; step={:d};h={:4.2f}".format(beta,omega,step,h))
    plt.tick_params(
        axis='x',         # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.ylim(0.01, 1000)
    plt.ylim(0.00, 1.001)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t)$")
    plt.savefig("corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.pdf".format(beta,omega,step,h),bbox_inches = 'tight')
    
def print_time(start_time):
    import datetime
    end = datetime.datetime.now()
    print("Start:",start_time)
    print("End",end)
    print("Total running time:",end-start_time)

if __name__ == "__main__":
    beta_list = [0.4,0.85,1.39,2.2]
    step = 13
    omega = 0.3
    h = 0.01
    for i,beta in enumerate(beta_list):
        plot_c_log(beta,omega,step,h) 
