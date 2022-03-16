#=======================================================
# The code main_ave_h calculate average population function for each initial configuration.
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main_ave_h.py  Gang Huang         2021-12-22    
#======
#Module
#======
import datetime
from scipy.stats import norm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from time import sleep

import multiprocessing as mp
from itertools import combinations 
import math
from random import randrange
from scipy.special import comb
import scipy as sp
from utilities_east_model import sample_step_list
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
skipped_steps = 0
num_steps = 0
#=========
#Functions
#=========
def ave_population(init,beta=beta,N=N,tot_steps=tot_steps,skiprows=skipped_steps):
    """0. Calculate <n_i>.
       1. We assume that <n_i> not depend on i.
    """
    ave5 = 0.0
    ave4 = 0.0
    ave3 = 0.0
    ave2 = 0.0
    ave1 = 0.0
    ave = 0.0
    s = int(tot_steps * 0.1) 
    skiprows = skipped_steps * N
    tot_steps2 = tot_steps - skipped_steps

    # Load data from a text file.
    fname = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps)
    # Read the j-th column as an array
    h = np.loadtxt(fname, dtype='float32', comments='#', delimiter=" ", converters=None, skiprows=skiprows, usecols=np.arange(0,N), ndmin=1, encoding='bytes')
    # ndmin: The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. Legal values: 0 (default), 1 or 2.
    # Calculate the self-auto correlation
    ave5 = ave5 + np.sum(h[:-5*N*s])
    ave4 = ave4 + np.sum(h[:-4*N*s])
    ave3 = ave3 + np.sum(h[:-3*N*s])
    ave2 = ave2 + np.sum(h[:-2*N*s])
    ave1 = ave1 + np.sum(h[:-N*s])
    ave = np.sum(h)
    norm = N*(tot_steps2)*N
    norm1 = N*(tot_steps2-s)*N
    norm2 = N*(tot_steps2-2*s)*N
    norm3 = N*(tot_steps2-3*s)*N
    norm4 = N*(tot_steps2-4*s)*N
    norm5 = N*(tot_steps2-5*s)*N
    ave5 = ave5/norm5
    ave4 = ave4/norm4
    ave3 = ave3/norm3
    ave2 = ave2/norm2
    ave1 = ave1/norm1
    ave = ave/norm
    print("ave5={}".format(ave5))
    print("ave4={}".format(ave4))
    print("ave3={}".format(ave3))
    print("ave2={}".format(ave2))
    print("ave1={}".format(ave1))
    print("ave={}".format(ave))
    f_log_corr = open('../data/ave_population_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    f_log_corr.write("# The averaged population operator over all nodes and for an initial configuration. (There are N nodes, num_cores initial configurations.)."+ "\n") 
    f_log_corr.write("{:8.7f}\n".format(ave5))
    f_log_corr.write("{:8.7f}\n".format(ave4))
    f_log_corr.write("{:8.7f}\n".format(ave3))
    f_log_corr.write("{:8.7f}\n".format(ave2))
    f_log_corr.write("{:8.7f}\n".format(ave1))
    f_log_corr.write("{:8.7f}\n".format(ave))
    f_log_corr.close()
    return ave5, ave4, ave3, ave2, ave1, ave
    #return ave

def calc_max_index(N,steps):
    for i in range(1,1200):
        max_step = sample_step_list(i)[-1]
        if max_step < steps * N:
           pass
        else:
           return i-1

#======
# tools
#======
def zero2minus1(arr):
    """Convert all 0 in arr to -1, while keep 1 not changed."""
    return (arr-1/2)*2.0

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
    beta,N,tot_steps = args.B,args.N,args.S
    skipped_steps = int(tot_steps * 0.2)
    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    param_tuple = [(i, beta,N,tot_steps,skipped_steps) for i in range(num_cores)]
    # Calculate the average population <n_i>.
    ave_h = [] 

    #Do NOT use mp HERE.
    for k in range(num_cores):
        ave_h.append(ave_population(k,beta,N,tot_steps,skipped_steps)) 
    mean_ave4_h = 0
    mean_ave3_h = 0
    mean_ave2_h = 0
    mean_ave1_h = 0
    mean_ave_h = 0
    for i in range(num_cores):
        mean_ave4_h = mean_ave4_h + ave_h[i][-5]
        mean_ave3_h = mean_ave3_h + ave_h[i][-4]
        mean_ave2_h = mean_ave2_h + ave_h[i][-3]
        mean_ave1_h = mean_ave1_h + ave_h[i][-2]
        mean_ave_h = mean_ave_h + ave_h[i][-1]
        #mean_ave_h = mean_ave_h + ave_h[i]
    mean_ave4_h = mean_ave4_h/num_cores
    mean_ave3_h = mean_ave3_h/num_cores
    mean_ave2_h = mean_ave2_h/num_cores
    mean_ave1_h = mean_ave1_h/num_cores
    mean_ave_h = mean_ave_h/num_cores
    #log
    f_log_mean_ave = open('../data/mean_ave_h_N{:d}_beta{:4.2f}_step{:d}.dat'.format(N,beta,tot_steps), "a+") 
    f_log_mean_ave.write("# The averaged h over all nodes (There are N nodes totally)."+ "\n") 
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave4_h))
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave3_h))
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave2_h))
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave1_h))
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave_h))
    f_log_mean_ave.close()


    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS DONE!")

