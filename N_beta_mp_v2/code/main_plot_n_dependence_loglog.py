#=======================================================
# The code main_corr calculate average corrlation function over all nodes for each initial configuration.
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main_corr.py  Gang Huang         2021-12-22    
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
N = 0 
tot_steps = 0 
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()

#=========
#Functions
#=========
def ave_population(init,beta=beta,N=N,tot_steps=tot_steps):
    """0. Calculate <n_i>.
       1. We assume that <n_i> not depend on i.
    """
    ave = 0.0
    skiped_steps = 200
    for j in range(N):
        # Load data from a text file.
        fname = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps)
        # Read the j-th column as an array
        h = np.loadtxt(fname, dtype='float32', comments='#', delimiter=" ", converters=None, skiprows=skiped_steps, usecols=j, ndmin=1, encoding='bytes')
        # Calculate the self-auto correlation
        ave = ave + np.sum(h)
    ave = ave/(N*(tot_steps-skiped_steps))
    print("ave={}".format(ave))
    f_log_corr = open('../data/ave_population_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    f_log_corr.write("# The averaged population operator over all nodes and for an initial configuration. (There are N nodes, num_cores initial configurations.)."+ "\n") 
    f_log_corr.write("{:8.7f}\n".format(ave))
    f_log_corr.close()
    return ave

def autocorr(init,ave_population,beta=beta,N=N,tot_steps=tot_steps):
    """1. save autocorr in an array.
       2. The returned auto correlation is the average over all nodes for a single initial configuration of the system..
    """
    corr = np.zeros(2*tot_steps-1)
    norm_arr  = 0.0
    for j in range(N):
        # Load data from a text file.
        fname = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps)
        # Read the j-th column as an array
        h = np.loadtxt(fname, dtype='float32', comments='#', delimiter=" ", converters=None, skiprows=0, usecols=j, ndmin=1, encoding='bytes')
        #print(h)
        # Calculate the self-auto correlation
        corr = corr + np.correlate(h-ave_population, h-ave_population, mode='full') # For small array only
        norm_arr = norm_arr + (h - ave_population)**2
    ave_corr = corr[corr.size//2:] / N # Average over nodes
    ave_corr = ave_corr / tot_steps # Average over starting points
    norm = np.sum(norm_arr) / (N*tot_steps)  # Average over all nodes
    ave_corr = ave_corr / norm # From the definition of the correlation function.

    print("ave_corr:{}".format(ave_corr))
    #Print corr
    np.save('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,tot_steps),ave_corr)
    f_log_corr = open('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    f_log_corr.write("# The averaged correlation over all nodes (There are N nodes totally)."+ "\n") 
    for i in range(tot_steps):
        f_log_corr.write("{:8.7f}\n".format(ave_corr[i]))
    f_log_corr.close()

class Spin_chain_east:
    def __init__(self,num_nodes):
        self.N = int(num_nodes) # fix it when I run the program.
        # Randomly set the initial coordinates,whose components are 1 or 0. 
        # To implement this, generate a uniform random sample from np.arange(2) of size num_nodes:
        np.random.seed()
        self.coord = np.random.choice(2, self.N) 
 
def plot_multiple_corr_loglog(corr1,corr2,corr3,init,beta=beta,N=N,tot_steps=tot_steps):
    time_arr = np.arange(tot_steps) 
    fig = plt.figure()
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(time_arr[1:-2],corr1[1:-2],label=r"$N=32$")
    ax.plot(time_arr[1:-2],corr2[1:-2],label="$64$")
    ax.plot(time_arr[1:-2],corr3[1:-2],label="$128$")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("init={:d}; beta={:3.1f}".format(init,beta))
    #
    #plt.savefig("../imag/Corr_init{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(init,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/N_dependence_of_corr_init{:d}_beta{:2.1f}_step{:d}_loglog.pdf".format(init,beta,tot_steps),format='pdf')

# ======
# Main
# ======
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
    beta,N,tot_steps = args.B,args.N,args.S
    
    N1 = 64; N2 = 128
    # Plot
    for k in range(16):
        for init in range(k*num_cores, (k+1)*num_cores):
            corr1 = np.load('../data/corr_N{:d}_beta{:3.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,tot_steps))     
            corr2 = np.load('../../N{:d}_beta{:3.1f}_mp/data/corr_N{:d}_beta{:3.2f}_init{:d}_step{:d}.npy'.format(N1,beta,N1,beta,init,tot_steps))     
            corr3 = np.load('../../N{:d}_beta{:3.1f}_mp/data/corr_N{:d}_beta{:3.2f}_init{:d}_step{:d}.npy'.format(N2,beta,N2,beta,init,tot_steps))     
            plot_multiple_corr_loglog(corr1,corr2,corr3,init,beta=beta,N=N,tot_steps=tot_steps)

    #Main 
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS PLOTTED!")

