#=======================================================
# The code main_corr calculate average corrlation function over all nodes for each initial configuration.
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main_n_dependence.py  Gang Huang         2021-12-22    
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
beta = 2.0 
N_list = [0]
N = 0 
tot_steps = 0 
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()

#=========
#Functions
#=========
def plot_multiple_corr(corr_arr,init,leng,beta=beta,N_list=N_list,steps=tot_steps):
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i,term in enumerate(N_list):
        max_index = calc_max_index(term,steps)                               
        step_list_v2 = sample_step_list(max_index)  
        time_arr = np.array(step_list_v2)/term
        ax.plot(time_arr[1:leng],corr_arr[i][:-1],label=r"$N=${:d}".format(term))
    ax.set_xscale('log')
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title(r"$\beta$={:4.2f}".format(beta))
    plt.savefig("../imag/N_dependence_of_corr_beta{:4.2f}_step{:d}.pdf".format(beta,steps),format='pdf')

# ======
# Main
# ======
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the number of nodes.") 
    parser.add_argument('-S', nargs='?', const=S, type=int, default=S, \
                        help="the number of total steps.")
    args = parser.parse_args()
    beta,tot_steps = args.B,args.S
    
    N_list = [64,128,256]
    leng = 29  # leng is the number of sampling points.
    skipped_steps = int(tot_steps * 0.2)
    tot_steps2 = tot_steps - skipped_steps
    # Plot
    corr0 = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(N_list[0],beta,tot_steps2))     
    corr_arr = np.zeros((len(N_list),leng))
    for i,term in enumerate(N_list):
        temp = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(term,beta,tot_steps2))     
        corr_arr[i] = temp[:leng]
    plot_multiple_corr(corr_arr,init,leng,beta=beta,N_list=N_list,steps=tot_steps2)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")

