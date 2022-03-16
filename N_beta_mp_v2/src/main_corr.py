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
from scipy.stats import norm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
init = 0
S = 0
num_cores = int(mp.cpu_count()) 
start_t = datetime.datetime.now()
skipped_steps = 0
num_steps = 0
Nstep = 10000

#=========
#Functions
#=========
def ave_population(init,beta=beta,N=N,tot_steps=Nstep,frac=0.8,skiprows=0):
    """0. Calculate <n_i>.
       1. We assume that <n_i> not depend on i.
    """
    ave = 0.0
    tot_steps2 = int(tot_steps * frac)
    # Load data from a text file.
    fname = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps)
    # Read the j-th column as an array
    h = np.loadtxt(fname, dtype='int8', comments='#', delimiter=" ", converters=None, skiprows=skiprows, usecols=np.arange(0,N), ndmin=1, encoding='bytes')
    # ndmin: The returned array will have at least ndmin dimensions. Otherwise mono-dimensional axes will be squeezed. Legal values: 0 (default), 1 or 2.
    # Calculate the self-auto correlation
    ave = np.sum(h)
    norm = N*(tot_steps2)*N
    ave = ave/norm
    f_log_corr = open('../data/ave_population_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    f_log_corr.write("# The averaged population operator over all nodes and for an initial configuration. (There are N nodes, num_cores initial configurations.)."+ "\n") 
    f_log_corr.write("{:8.7f}\n".format(ave))
    f_log_corr.close()
    return ave

def autocorr_3(init,beta=beta,N=N,tot_steps=Nstep,frac=0.8):
    """1. save autocorr in an array.
       2. The returned auto correlation is the average over all nodes for a single initial configuration of the system..
    """
    corr = 0 
    norm_arr = 0.0
    fname = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps)
    df = np.loadtxt(fname, skiprows=0, encoding='bytes')
    df_bar = df.mean()
    for j in range(N):
        # Load data from a text file.
        # Read the j-th column as an array
        # Calculate the self-auto correlation
        h = df[:,j]
        corr = corr +  my_autocorr_simple(h,df_bar,tot_steps,frac)
    ave_corr = corr/N # Average over nodes
    #Print corr
    tot_steps2 = int(tot_steps * frac)
    np.save('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,tot_steps), ave_corr)
    f_log_corr = open('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    f_log_corr.write("# The averaged correlation over all nodes (There are N nodes totally)."+ "\n") 
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    for i, term  in enumerate(step_list_v2):
        f_log_corr.write("{:d} {:8.7f}\n".format(term, ave_corr[i]))
    f_log_corr.close()

def my_autocorr(df,tot_steps=Nstep,frac=0.8):
    ''' Ref: https://towardsdatascience.com/understanding-autocorrelation-in-time-series-analysis-322ad52f2199
    Returns autocorrelation coefficient for lags [nlags, 0]
    df: dataframe
        Input dataframe of time series
    nlags: int
           maximum number of lags, default 2
    Returns
    array: autocorrelation coefficients for lags [nlags, 0]
    '''
    def autocorr(y, lag=2):
        '''
        Calculates autocorrelation coefficient for single lag value
        y: array
           Input time series array
        lag: int, default: 2 
             'kth' lag value
        Returns
        int: autocorrelation coefficient 
        '''
        y = np.array(y).copy()
        y_bar = np.mean(y) #y_bar = mean of the time series y
        denominator = sum((y - y_bar) ** 2) #sum of squared differences between y(t) and y_bar
        eps = 0.000001
        if denominator < eps: denominator = eps
        numerator_p1 = y[lag:] - y_bar #y(t)-y_bar: difference between time series (from 'lag' till the end) and y_bar
        numerator_p2 = y[:-lag] - y_bar #y(t-k)-y_bar: difference between time series (from the start till lag) and y_bar
        numerator = sum(numerator_p1 * numerator_p2) #sum of y(t)-y_bar and y(t-k)-y_bar
        return (numerator / denominator)
    tot_steps2 = int(tot_steps * frac)
    print("steps2:{}".format(tot_steps2))
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    acf = [1] #initializing list with autocorrelation coefficient for lag k=0 which is always 1
    for i in step_list_v2: 
        acf.append(autocorr(df[:], lag=i)) #calling autocorr function for each lag 'i'
    return np.array(acf)

def my_autocorr_simple(df,df_bar,tot_steps=Nstep, frac=0.8):
    ''' Ref: https://towardsdatascience.com/understanding-autocorrelation-in-time-series-analysis-322ad52f2199
    Returns autocorrelation coefficient for lags [nlags, 0]
    df: dataframe
        Input dataframe of time series
    nlags: int
           maximum number of lags, default 2
    Returns
    array: autocorrelation coefficients for lags [nlags, 0]
    # y_bar = mean of the time series y. We use mean of df (df_bar) to represent y_bar
    '''
    def autocorr(y, df_bar, lag=2):
        '''
        Calculates autocorrelation coefficient for single lag value
        y: array
           Input time series array
        lag: int, default: 2 
             'kth' lag value
        Returns
        int: autocorrelation coefficient 
        '''
        y = np.array(y).copy()
        y_bar = df_bar
        denominator = sum((y - y_bar) ** 2) #sum of squared differences between y(t) and y_bar
        eps = 0.000001
        if denominator < eps: denominator = eps
        numerator_p1 = y[lag:] - y_bar #y(t)-y_bar: difference between time series (from 'lag' till the end) and y_bar
        numerator_p2 = y[:-lag] - y_bar #y(t-k)-y_bar: difference between time series (from the start till lag) and y_bar
        numerator = sum(numerator_p1 * numerator_p2) #sum of y(t)-y_bar and y(t-k)-y_bar
        return (numerator / denominator)
    tot_steps2 = int(tot_steps * frac)
    print("steps2:{}".format(tot_steps2))
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    acf = [1] #initializing list with autocorrelation coefficient for lag k=0 which is always 1
    for i in step_list_v2: 
        acf.append(autocorr(df[:],df_bar, lag=i)) #calling autocorr function for each lag 'i'
    return np.array(acf)

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
#def zero2minus1(arr):
#    """Convert all 0 in arr to -1, while keep 1 not changed."""
#    return (arr-1/2)*2.0
def plot_corr(corr,init,step_list_v2,beta=beta,N=N):
    time_arr = np.array(step_list_v2)/N 
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((0.001,1))
    num_color = 5 
    #color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[:],corr[:-1],marker='.',label="correl")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C(t)$")
    ax.set_title("init={:d};N={:d}; beta={:3.1f}".format(init,N,beta))
    #
    plt.savefig("../imag/corr_init{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(init,N,beta,tot_steps2),format='pdf')

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
    beta,N, Nstep = args.B,args.N, args.S
    frac = 0.8
    tot_steps2 = int(Nstep * frac)
    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    param_tuple = [(i, beta,N,Nstep,0) for i in range(num_cores)]
    # Calculate the average population <n_i>.
    ave_h = [] 

    #Do NOT use mp HERE.
    for k in range(num_cores):
        ave_h.append(ave_population(k,beta,N,Nstep,frac=frac)) 
    mean_ave_h = 0
    for i in range(num_cores):
        mean_ave_h = mean_ave_h + ave_h[i]
    mean_ave_h = mean_ave_h/num_cores
    #log
    f_log_mean_ave = open('../data/mean_ave_h_N{:d}_beta{:4.2f}_step{:d}.dat'.format(N,beta,Nstep), "a+") 
    f_log_mean_ave.write("# The averaged h over all nodes (There are N nodes totally)."+ "\n") 
    f_log_mean_ave.write('{:8.6f}\n'.format(mean_ave_h))
    f_log_mean_ave.close()
    # Calcuate correlation function
    param_tuple1 = [(i, beta,N,Nstep,0.8) for i in range(num_cores)]
    for k in range(num_cores):
        print("Now start process ({}).".format(k))
        mp.Process(target=autocorr_3, args=param_tuple1[k]).start() #start now
    # The mean correlation function
    mean_corr = 0

    # Plot
    for init in range(num_cores):
        corr = np.load('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,tot_steps2))
        mean_corr = mean_corr + corr
    mean_corr = mean_corr / num_cores
    np.save('../data/mean_corr_N{:d}_beta{:4.2f}_step{:d}.npy'.format(N,beta,tot_steps2),mean_corr)
    
    max_index = calc_max_index(N,tot_steps2)                               
    step_list_v2 = sample_step_list(max_index)  
    #IF YOU WANT TO PLOT FOR EACH CORE, USE THE FOLLOWING 3 LINES
    #for init in range(num_cores):
    #    corr = np.load('../data/corr_N{:d}_beta{:4.2f}_init{:d}_step{:d}.npy'.format(N,beta,init,tot_steps2))     
    #    plot_corr(corr,init,step_list_v2,beta=beta,N=N)

    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("AUTO CORRELATIONS DONE!")

