#=======================================================
# The code main_beta_dependece calculate the beta dependece of the average corrlation function.
#=======================================================
# Function                 Author                 Date
# =======                 ========            ==========
# main_beta_dependece.py  Gang Huang          2022-3-14    
#======
#Module
#======
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import math
from random import randrange

#===========
# Parameters
#===========
beta = 0.0 
beta_list = [0.0]
c_list = [0.0]
step = 13
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()
h = 0.04 # Set the value of h is the same as that in c_EJ.py
omega = 0.0

#=========
#Functions
#=========
def plot_multiple_corr(corr_arr,beta_list=beta_list,c_list=c_list,omega=omega,steps=step,h=0.08):
    time_arr = [i*h for i in range(corr_arr.shape[1])]
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_ylim(0.01, 1.0)
    ax.set_xlim(0.1, 3000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, term in enumerate(c_list):
        ax.plot(time_arr[1:],corr_arr[i][1:],label=r"$c=${}".format(term))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title(r"Eisinger-Jaeckle Approx:$\omega=${:4.2f},$h=${:4.2f}.".format(omega,h))
    plt.savefig("../imag/beta_dependence_of_corr_omega{:4.2f}_step{:d}_h{:4.2f}_loglog.pdf".format(omega,step,h),format='pdf')


# ======
# Main
# ======
if __name__=='__main__':
    import argparse
    h = 0.04
    omega=0.3
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', nargs='?', const=h, type=float, default=h, \
                        help="the step size.")
    parser.add_argument('-W', nargs='?', const=1, type=float, default=1, \
                        help="the number of nodes per layer.")
    args = parser.parse_args()
    h, omega = args.H, args.W

    step = 13
    beta_list = [ 0.40, 0.85, 1.39, 2.2]
    c_list = [0.4, 0.3, 0.2, 0.1]
    #beta_list = [-2.2, -1.39, -0.85, -0.40]
    #c_list = [0.9, 0.8, 0.7, 0.6]
    # Plot
    corr0 = np.load('../data/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta_list[0],omega,step,h))     
    corr_arr = np.zeros((len(beta_list),corr0.shape[0]))
    for i,term in enumerate(beta_list):
        temp = np.load('../data/corr_EJ_beta{:4.2f}_omega{:4.2f}_step{:d}_h{:4.2f}.npy'.format(term,omega,step,h))     
        corr_arr[i] = temp
    plot_multiple_corr(corr_arr,beta_list=beta_list,c_list=c_list,omega=omega,steps=step,h=h)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")
