#=======================================================
# The code main_corr calculate average corrlation function over all nodes for each initial configuration.
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main_plot_corr.py  Gang Huang         2021-12-22    
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
N = 0 
tot_steps = 0 
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()
skipped_steps = 10

#=========
#Functions
#=========
def plot_corr(corr,init,beta=beta,N=N,steps=tot_steps,frac=0.8):
    tot_steps2 = int(steps* frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_xscale('log')
    ax.plot(time_arr[1:],corr[1:-1],label="correl")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("init={:d};N={:d}; beta={:4.2f}".format(init,N,beta))
    #
    plt.savefig("../imag/corr_init{:d}_N{:d}_beta{:4.2f}_steps{:d}_s.pdf".format(init,N,beta,steps),format='pdf')

def plot_mean_corr(corr,beta=beta,N=N,steps=tot_steps,frac=0.8):
    tot_steps2 = int(steps* frac)
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    time_arr = np.array(step_list_v2)/N
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_xscale('log')
    ax.plot(time_arr[1:],corr[1:-1],label="Ave Correl")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("N={:d}; beta={:3.1f}".format(N,beta))
    #
    plt.savefig("../imag/Mean_corr_N{:d}_beta{:4.2f}_steps{:d}.pdf".format(N,beta,steps),format='pdf')

# ====
# Main
# ====
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-N', nargs='?', const=N, type=int, default=N, \
                        help="the number of nodes.") 
    parser.add_argument('-S', nargs='?', const=S, type=int, default=S, \
                        help="the number of total steps.")
    args = parser.parse_args()
    beta,N,Nstep = args.B,args.N,args.S
    
    # Plot
    #skiprows = skipped_steps * N
    frac = 0.8
    tot_steps2 = int(Nstep* frac)
    mean_corr = 0 
    for k in range(16):
        for init in range(k*num_cores, (k+1)*num_cores):
            corr = np.load('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,Nstep))
            mean_corr = mean_corr + corr
    mean_corr = mean_corr / 64
    np.save('../data/mean_corr_N{:d}_beta{:4.2f}_step{:d}.npy'.format(N,beta,Nstep),mean_corr)
    ## Plot
    corr = np.load('../data/mean_corr_N{:d}_beta{:4.2f}_step{:d}.npy'.format(N,beta,Nstep))     
    plot_mean_corr(corr,beta=beta,N=N,steps=Nstep,frac=frac)

    # Write the mean correlation function into dat file
    f_corr = open('../data/mean_corr_N{:d}_beta{:4.2f}_step{:d}.dat'.format(N,beta,Nstep), "w") 
    f_corr.write("# The averaged correlation over all nodes (There are N nodes totally)."+ "\n") 
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    for i, term  in enumerate(step_list_v2):
        f_corr.write("{:d} {:8.7f}\n".format(term, corr[i]))
    f_corr.close()

    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS DONE!")
