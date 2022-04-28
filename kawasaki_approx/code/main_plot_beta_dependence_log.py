#=======================================================
# The code main_beta_dependece calculate the beta dependece of the average corrlation function.
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main_beta_dependece.py  Gang Huang         2021-12-22    
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
h = 0.08

#=========
#Functions
#=========
def plot_multiple_corr(corr_arr,beta_list=beta_list,c_list=c_list,steps=step,h=0.08):
    time_arr = [i*h for i in range(corr_arr.shape[1])]
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.0, 1.0)
    ax.set_xscale('log')
    for i, term in enumerate(c_list):
        ax.plot(time_arr[1:],corr_arr[i][1:],label=r"$c=${}".format(term))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title(r"Kawasaki Approx: $h=${:4.2f}.".format(h))
    plt.savefig("../imag/beta_dependence_of_corr_step{:d}_h{:4.2f}_log.pdf".format(step,h),format='pdf')
    plt.savefig("../imag/beta_dependence_of_corr_step{:d}_h{:4.2f}_log.png".format(step,h),format='png')


# ======
# Main
# ======
if __name__=='__main__':
    h = 0.04
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', nargs='?', const=h, type=float, default=h, \
                        help="the step size.")
    args = parser.parse_args()
    h = args.H

    step = 13
    beta_list = [0.40,0.85,1.39,2.2]
    c_list = [0.4,0.3,0.2,0.1]
    # Plot
    corr0 = np.load('../data/corr_K_beta{:4.2f}_step{:d}_h{:4.2f}.npy'.format(beta_list[0],step,h))     
    corr_arr = np.zeros((len(beta_list),corr0.shape[0]))
    for i,term in enumerate(beta_list):
        temp = np.load('../data/corr_K_beta{:4.2f}_step{:d}_h{:4.2f}.npy'.format(term,step,h))     
        corr_arr[i] = temp
    plot_multiple_corr(corr_arr,beta_list=beta_list,c_list=c_list,steps=step,h=h)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")

