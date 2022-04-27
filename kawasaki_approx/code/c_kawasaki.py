#==========================================================
# Name: c_kawasaki.py
# Function: this code is designed for calculate correlation
# functions numerically for 1D East models.
# In this code, we implemente the solving of an integro-
# differential eqn. with initial conditions.
# Rf.: notes_SVD_glass.pdf 
#=========================
# Author: Gang Huang 
# Data: 2022-3-14
# Version: 1     
#=========================
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_log(beta,omega,step_exp,h=0.01):
    c = np.load('../data/corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step_exp,h))
    print("C=",c)
    length = c.shape[0]
    t = np.array([x for x in range(length)])
    fig = plt.figure()
    #plt.xscale('log')
    #plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.plot(t*h, c, "-", color="black",)
    ax.set_title(r"$\beta$={:4.2f}; $\omega ={:4.2f}$".format(beta,omega))
    plt.ylim(0.01, 1.0)
    plt.xlim(0.1, 3000)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$C(t)$")
    plt.savefig("../imag/corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.pdf".format(beta,omega,step_exp,h),bbox_inches = 'tight')

def print_time(start_time):
    import datetime
    end = datetime.datetime.now()
    print("Start:",start_time)
    print("End",end)
    print("Total running time:",end-start_time)

if __name__ == "__main__":
    import datetime
    #Time start
    start = datetime.datetime.now()

    # Parameters and initial conditions.
    h = 0.04
    omega = 0.1

    h_squard = h ** 2
    step_exp = 13
    n_tot = 2**step_exp

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', nargs='?', const=h, type=float, default=h, \
                        help="the step size.")
    parser.add_argument('-W', nargs='?', const=1, type=float, default=1, \
                        help="the number of nodes per layer.")
    args = parser.parse_args()
    h, omega = args.H, args.W

    beta_list = [-2.2, -1.39, -0.85, -0.40, 0.08, 0.40, 0.85, 1.39, 2.2]
    for i, beta in enumerate(beta_list):
        mean_occ = np.exp(-beta)/(1+np.exp(-beta))
        c = np.ones((n_tot)) * np.nan 
        a2 = mean_occ * (1-mean_occ)
        # Initial conditions
        for i in range(n_tot):  
            c[i] = 1
        c[0] = 1

        c[1] =(1-h*omega)*c[0]
        # Start to calcualte C, iteratively.  
        for n in range(1,n_tot-1):
            temp_arr = np.zeros(n)
            for n1 in range(1,n):
                temp_arr[n1] = h * a2 * (c[n-n1]**2) *(c[n1]-c[n1-1]) 
            dc = -np.sum(temp_arr)/omega
            dc2 = (1-h*omega) * c[n]
            #dc2 = h*(1-omega) * c[n]
            c[n+1] = dc + dc2

        print("c:",c)
        filename_c = "../data/corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.dat".format(beta,omega,step_exp,h)
        with open(filename_c, 'w') as f: # able to append data to file
            for s in range(n_tot):
                f.write(" ".join([str(s*h),str(c[s]),'\n'])) 
            f.close() # You can add this but it is not mandatory
        #Write C(t) 
        np.save('../data/corr_K_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta,omega,step_exp,h),c)
        
        #Plot correlation function
        plot_corr_log(beta,omega,step_exp,h=h)

    # Print the total time used
    end = datetime.datetime.now()
    print("Start:",start)
    print("End",end)
    print("Total running time (ElementWise Operations):",end-start)
