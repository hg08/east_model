#=======================================================
# The code main_east does MC simulations
#=======================================================
# Function       Author              Date
# =======        ========           ==========
# p1_east_mc.py  Gang Huang         2021-12-22    
# AIM: For each core, print the MC trajectory of the spin chain on the logfile (f_log_sigma).
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
 
    def update(self,beta,f_log_sigma):
        """One step, at most one node moves."""
        for _ in range(N):
            i = np.random.choice(N) # take one integer randomly from 0 to N-1.
            if i < self.N-1:
                # If the (i+1)-th node is equals to 0, then keep it.
                if self.coord[i+1] == 0: 
                    self.record_coord(f_log_sigma)
                else:
                    if self.coord[i] == 1:
                        self.coord[i] = 0 
                        self.record_coord(f_log_sigma)
                    else:
                        prob = np.random.random(1)
                        if prob < np.exp(-beta): 
                            self.coord[i] = 1
                        self.record_coord(f_log_sigma)
            else:
                # If the (i+1)-th node is equals to 0, then keep it.
                if self.coord[np.mod((i+1),self.N)] == 0: 
                    self.record_coord(f_log_sigma)
                else:
                    if self.coord[i] == 1:
                        self.coord[i] = 0 
                        self.record_coord(f_log_sigma)
                    else:
                        prob = np.random.random(1)
                        if prob < np.exp(-beta): 
                            self.coord[i] = 1
                        self.record_coord(f_log_sigma)

    def record_coord(self, f_log_sigma):
        string_list = ["%i" % value for value in self.coord]
        formatted_string = " ".join(string_list)
        f_log_sigma.write(formatted_string + "\n") # https://www.kite.com/python/answers/how-to-print-a-list-using-a-custom-format-in-python

def mc(N=N,tot_steps=tot_steps,beta=beta,init=init):
    f_log_sigma = open('../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+")   # Open log
    r = Spin_chain_east(N) # Initilize an instance of a spin chain.
    for i in range(tot_steps):
        r.update(beta,f_log_sigma) # Print the new coord as a record in f_log_sigma
    f_log_sigma.close() # Close log
         
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
    
    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    param_tuple = [(N,tot_steps,beta,i) for i in range(num_cores)]

    #MC simulations for different init configurations
    for k in range(num_cores):
        print("Now start process ({}).".format(k))
        mp.Process(target=mc, args=param_tuple[k]).start() #start now

    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("DONE!")

