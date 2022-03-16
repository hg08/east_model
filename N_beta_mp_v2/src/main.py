#=======================================================
# The code main_east does MC simulations
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# main.py  Gang Huang         2021-12-22    
# AIM: For each core, print the MC trajectory of the spin chain on the logfile (f_log_sigma).
# In version 2, we let the total number as a function of beta, to make sure for each beta, we can obtain the configurations at equlibrium.
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
import os
from random import randrange
from scipy.special import comb

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
class Spin_chain_east:
    def __init__(self,num_nodes):
        self.N = int(num_nodes) # fix it when I run the program.
        # Randomly set the initial coordinates,whose components are 1 or 0. 
        # To implement this, generate a uniform random sample from np.arange(2) of size num_nodes:
        np.random.seed() #Not set the argument, the function seed() will use the time as the argument.
        self.coord = np.random.choice(2, self.N) 
 
    def update(self,beta,f_log):
        """One step, at most one node moves."""
        for _ in range(N):
            i = np.random.choice(N) # take one integer randomly from 0 to N-1.
            if i < self.N-1:
                # If the (i+1)-th node is equals to 0, then keep it.
                if self.coord[i+1] == 0: 
                    self.record_coord(f_log)
                else:
                    if self.coord[i] == 1:
                        self.coord[i] = 0 
                        self.record_coord(f_log)
                    else:
                        prob = np.random.random(1)
                        if prob < np.exp(-beta): 
                            self.coord[i] = 1
                        self.record_coord(f_log)
            else:
                # If the (i+1)-th node is equals to 0, then keep it.
                if self.coord[np.mod((i+1),self.N)] == 0: 
                    self.record_coord(f_log)
                else:
                    if self.coord[i] == 1:
                        self.coord[i] = 0 
                        self.record_coord(f_log)
                    else:
                        prob = np.random.random(1)
                        if prob < np.exp(-beta): 
                            self.coord[i] = 1
                        self.record_coord(f_log)

    def record_coord(self, f_log):
        string_list = ["%i" % value for value in self.coord]
        formatted_string = " ".join(string_list)
        f_log.write(formatted_string + "\n") # https://www.kite.com/python/answers/how-to-print-a-list-using-a-custom-format-in-python

def LastNlines2new(fname_str, fnewname_str,n):
    # opening file using with() method
    # so that file get closed
    # after completing work
    f_sigma = open(fnewname_str, "w")   # Open sigmafile
    with open(fname_str) as file:
        # loop to read iterate
        # last n lines and print it
        for line in (file.readlines() [-n:]):
            f_sigma.write(line) 
    f_sigma.close() # Close log

def mc(N=N,tot_steps=tot_steps,beta=beta,init=init,Nstep=3000,rate=0.8):
    log_str = '../data/log_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,Nstep)
    sigma_str = '../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,Nstep)
    try:
        os.remove(log_str)
    except:
        pass
    try:
        os.remove(sigma_str)
    except:
        pass
    f_log = open(log_str, "w")   # Open log file
    r = Spin_chain_east(N) # Initilize an instance of a spin chain.
    for i in range(tot_steps):
        r.update(beta,f_log) # Print the new coord as a record in f_log_sigma
    # Only save the last N*Nstep*alpha (eg, 3000 * 0.8 steps)
    num_of_lines = int(N*Nstep*rate)
    LastNlines2new(log_str,sigma_str,num_of_lines) #cp useful data so that we do not need to skip lines when load data for calculating correlation function
    f_log.close() # Close log
    os.remove(log_str) # Delete log_str to save space and time; 
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
    
    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    tot_steps = int(Nstep*(1+beta)**2) # We have let the tot_step depends on beta !!
    rate = 0.8
    param_tuple = [(N,tot_steps,beta,i,Nstep,rate) for i in range(num_cores)]

    #MC simulations for different init configurations
    for k in range(num_cores):
        print("Now start process ({}).".format(k))
        mp.Process(target=mc, args=param_tuple[k]).start() #start now

    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("DONE!")

