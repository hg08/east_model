#========================================================
#Name: c_EJ.py 
#Function: this code is designed for calculate correlation
# functions numerically for 1D East models.
# In this code, we implemente the solving of an integro-
# differential eqn. with initial conditions.
# Rf.: notes_SVD_glass.pdf; Eisinger-Jackle1991,1993; 
#=========================
#Author: Gang Huang 
#Data: 2022-3-14
#Version: 1     
#=========================
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_log(beta,omega,step_exp,h=0.01):
    c = np.load('../data/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step_exp,h))
    print("C=",c)
    length = c.shape[0]
    t = np.array([x for x in range(length)])
    fig = plt.figure()
    #plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, c, "-", color="black",)
    ax.set_title(r"$\beta$={:4.2f}; $\omega ={:4.2f}$".format(beta,omega))
    plt.ylim(0.01, 1.001)
    plt.xlim(0.001, 1000)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t)$")
    plt.savefig("../imag/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.pdf".format(beta,omega,step_exp,h),bbox_inches = 'tight')

def print_time(start_time):
    import datetime
    end = datetime.datetime.now()
    print("Start:",start_time)
    print("End",end)
    print("Total running time:",end-start_time)

if __name__ == "__main__":
    """Time start"""
    import datetime
    start = datetime.datetime.now()

    """Parameters and initial conditions."""
    #final Temperature Tf and intial Temperature T0 =1/beta0
    h = 0.08
    h_squard = h ** 2
    omega = 0.5 # Q: 0.3 (or 0.4) is just a template value I chosen. How to calcualte omega?

    step_exp = 13
    n_tot = 2**step_exp

    beta_list = [0.08, 0.40, 0.85, 1.39, 2.00, 2.2]
    for i, beta in enumerate(beta_list):
        mean_occ = np.exp(-beta)/(1+np.exp(-beta))
        c = np.ones((n_tot)) * np.nan 
        a2 = mean_occ * (1-mean_occ)
        # Initial conditions
        for i in range(n_tot):  
            c[i] = 1
        c[0] = 1

        c[1] =(1-h*omega)*c[0]
        #Start to calcualte C, iteratively.  
        for n in range(1,n_tot-1):
            temp_arr = np.zeros(n)
            for n1 in range(1,n):
                temp_arr[n1] = h * a2 * c[n-n1]*(c[n1]-c[n1-1]) 
            dc = -np.sum(temp_arr)/omega
            dc2 = (1-h*omega) * c[n]
            c[n+1] = dc + dc2

        print("c:",c)
        filename_c = "../data/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.dat".format(beta,omega,step_exp,h)
        with open(filename_c, 'w') as f: # able to append data to file
            for s in range(n_tot):
                f.write(" ".join([str(s*h),str(c[s]),'\n'])) 
            f.close() # You can add this but it is not mandatory
        #write C(t) 
        np.save('../data/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step_exp,h),c)

        #Plot correlation function
        plot_corr_log(beta,omega,step_exp,h=h)

    # #show the total time used
    end = datetime.datetime.now()
    print("Start:",start)
    print("End",end)
    print("Total running time (ElementWise Operations):",end-start)
