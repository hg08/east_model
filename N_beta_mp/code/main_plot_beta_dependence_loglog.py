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
from utilities import calc_max_index, sample_step_list

#===========
# Parameters
#===========
beta = 0.0 
beta_list = [0.0]
N = 0 
tot_steps = 0 
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()

#=========
#Functions
#=========
def plot_multiple_corr(corr_arr,init,beta_list=beta_list,N=N,steps=tot_steps):
    max_index = calc_max_index(N,steps)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_ylim(0.01, 1.01)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, term in enumerate(beta_list):
        ax.plot(time_arr[1:],corr_arr[i][1:-1],label=r"$\beta=${}".format(term))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("N={:d}".format(N))
    #
    plt.savefig("../imag/beta_dependence_of_corr_N{:d}_step{:d}_loglog.pdf".format(N,tot_steps),format='pdf')

# ======
# Main
# ======
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', nargs='?', const=N, type=int, default=N, \
                        help="the number of nodes.") 
    parser.add_argument('-S', nargs='?', const=S, type=int, default=S, \
                        help="the number of total steps.")
    args = parser.parse_args()
    N,tot_steps = args.N,args.S
    
    beta_list = [0.05,0.2,0.3,0.4,1.0,2.0,3.0,4.0,5.0]
    skipped_steps = int(tot_steps * 0.2)
    tot_steps2 = tot_steps - skipped_steps
    # Plot
    corr0 = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(N,beta_list[0],tot_steps2))     
    corr_arr = np.zeros((len(beta_list),corr0.shape[0]))
    for i,term in enumerate(beta_list):
        temp = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(N,term,tot_steps2))     
        corr_arr[i] = temp
    plot_multiple_corr(corr_arr,init,beta_list=beta_list,N=N,steps=tot_steps2)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")

