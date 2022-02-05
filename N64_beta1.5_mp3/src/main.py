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
def line2intlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(int(x))
    return res_list
def line2floatlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(float(x))
    return res_list

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
    # Open log
    f_log_sigma = open('../data/sigma_N{:d}_beta{:4.2f}_init{:d}_step{:d}.dat'.format(N,beta,init,tot_steps), "a+") 
    # Initilize an instance of a spin chain.
    r = Spin_chain_east(N)
    for i in range(tot_steps):
        # Print the new coord as a record in f_log_sigma
        r.update(beta,f_log_sigma)

    # Close log
    f_log_sigma.close()
         
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
    param_tuple = [(N,tot_steps,beta,0),
                   (N,tot_steps,beta,1),
                   (N,tot_steps,beta,2),
                   (N,tot_steps,beta,3),
                   (N,tot_steps,beta,4),
                   (N,tot_steps,beta,5),
                   (N,tot_steps,beta,6),
                   (N,tot_steps,beta,7),
                   (N,tot_steps,beta,8),
                   (N,tot_steps,beta,9),
                   (N,tot_steps,beta,10),
                   (N,tot_steps,beta,11),
                   (N,tot_steps,beta,12),
                   (N,tot_steps,beta,13),
                   (N,tot_steps,beta,14),
                   (N,tot_steps,beta,15),
                   (N,tot_steps,beta,16),
                   (N,tot_steps,beta,17),
                   (N,tot_steps,beta,18),
                   (N,tot_steps,beta,19),
                   (N,tot_steps,beta,20),
                   (N,tot_steps,beta,21),
                   (N,tot_steps,beta,22),
                   (N,tot_steps,beta,23),
                   (N,tot_steps,beta,24),
                   (N,tot_steps,beta,25),
                   (N,tot_steps,beta,26),
                   (N,tot_steps,beta,27),
                   (N,tot_steps,beta,28),
                   (N,tot_steps,beta,29),
                   (N,tot_steps,beta,30),
                   (N,tot_steps,beta,31),
                   (N,tot_steps,beta,32),
                   (N,tot_steps,beta,33),
                   (N,tot_steps,beta,34),
                   (N,tot_steps,beta,35),
                   (N,tot_steps,beta,36),
                   (N,tot_steps,beta,37),
                   (N,tot_steps,beta,38),
                   (N,tot_steps,beta,39),
                   (N,tot_steps,beta,40),
                   (N,tot_steps,beta,41),
                   (N,tot_steps,beta,42),
                   (N,tot_steps,beta,43),
                   (N,tot_steps,beta,44),
                   (N,tot_steps,beta,45),
                   (N,tot_steps,beta,46),
                   (N,tot_steps,beta,47),
                   (N,tot_steps,beta,48),
                   (N,tot_steps,beta,49),
                   (N,tot_steps,beta,50),
                   (N,tot_steps,beta,51),
                   (N,tot_steps,beta,52),
                   (N,tot_steps,beta,53),
                   (N,tot_steps,beta,54),
                   (N,tot_steps,beta,55),
                   (N,tot_steps,beta,56),
                   (N,tot_steps,beta,57),
                   (N,tot_steps,beta,58),
                   (N,tot_steps,beta,59),
                   (N,tot_steps,beta,60),
                   (N,tot_steps,beta,61),
                   (N,tot_steps,beta,62),
                   (N,tot_steps,beta,63)]
    #MC simulations for different init configurations
    for k in range(num_cores):
        print("Now start process ({}).".format(k))
        mp.Process(target=mc, args=param_tuple[k]).start() #start now

    #Main MC simulations DONE
    print("The computer has " + str(num_cores) + " cores.")
    print("DONE!")

