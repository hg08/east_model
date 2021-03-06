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

def sample_step_list(step_index=30):
    # The parameter ratio is used for generating the sampling time point.
    q = 2.0
    p = q + 1
    ratio= p/q
    # general ratio
    step_list = [int((ratio)**n) for n in range(step_index)]
    return step_list

