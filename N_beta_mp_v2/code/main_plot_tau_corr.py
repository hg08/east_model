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
c_list = [0.0]
N = 0 
tot_steps = 0 
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()
Nstep = 10000

#=========
#Functions
#=========
def plot_multiple_corr(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep, frac=0.8):
    tot_steps2 = int(steps * frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_ylim(0.01, 1.01)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for i, term in enumerate(c_list):
        ax.plot(time_arr[1:],corr_arr[i][1:-1],label=r"$c=${}".format(term))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("N={:d}".format(N))
    #
    plt.savefig("../imag/beta_dependence_of_corr_N{:d}_step{:d}_loglog.pdf".format(N,steps),format='pdf')

def relaxation_time_index(mean_corr,N=N, steps=Nstep,frac=0.8):
    tot_steps2 = int(steps * frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)
    time_arr = np.array(step_list_v2)/N
    cutoff = 1/np.e
    size = mean_corr.shape[0]
    ndx = 0
    if mean_corr[-1] < cutoff:
        for i, term in enumerate(mean_corr):
            if term < cutoff:
                ndx = i-1 # Here, we use i-1, instead of i. So we have deliberately underestimated the relaxation time.
                break
    else:
        ndx = size-1
    return time_arr[ndx]

def plot_multiple_tau_of_beta(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep,frac=0.8):
    tot_steps2 = int(steps * frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_ylim(1.0, 10000)
    ax.set_yscale('log')
    tau_arr = np.zeros(len(beta_list))
    for i, term in enumerate(beta_list):
        tau_arr[i] = relaxation_time_index(corr_arr[i],N=N,steps=steps,frac=frac)
    ax.plot(beta_list,tau_arr,label="N={}".format(N),marker="o")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\tau(\beta)$")
    ax.set_title("Relaxation Time")
    #
    plt.savefig("../imag/relaxation_time_of_corr_on_beta_N{:d}_step{:d}.pdf".format(N,Nstep),format='pdf')

def plot_multiple_tau_of_c(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep,frac=0.8):
    tot_steps2 = int(steps * frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_ylim(1.0, 10000)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    tau_arr = np.zeros(len(c_list))
    for i, term in enumerate(c_list):
        tau_arr[i] = relaxation_time_index(corr_arr[i],N=N,steps=steps,frac=frac)
    ax.plot(c_list,tau_arr,label="N={}".format(N),marker='v')
    plt.legend(loc="lower left")
    plt.xlabel(r"$c$")
    plt.ylabel(r"$\tau(c)$")
    ax.set_title("Relaxation Time")
    #
    plt.savefig("../imag/relaxation_time_of_corr_on_c_N{:d}_step{:d}.pdf".format(N,steps),format='pdf')
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
    N,Nstep = args.N,args.S
    frac = 0.8
    beta_list = [0.08,0.40,0.85,1.39,2.0,2.2]
    c_list = [0.48,0.40,0.3,0.2,0.12,0.1]
    tot_steps2 = int(Nstep*frac)
    # Plot
    corr0 = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(N,beta_list[0],Nstep))     
    corr_arr = np.zeros((len(beta_list),corr0.shape[0]))
    for i,term in enumerate(beta_list):
        temp = np.load('../data/mean_corr_N{:d}_beta{:3.2f}_step{:d}.npy'.format(N,term,Nstep))     
        corr_arr[i] = temp
    plot_multiple_corr(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep,frac=frac)
    # Plot tau of beta
    plot_multiple_tau_of_beta(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep,frac=frac)
    # Plot tau of c
    plot_multiple_tau_of_c(corr_arr,init,beta_list=beta_list,c_list=c_list,N=N,steps=Nstep,frac=frac)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")

