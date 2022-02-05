import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import math
from random import randrange

def calc_max_index(N,tot_steps):
    # To obtain 
    for i in range(1,1200):
        max_step = sample_step_list(i)[-1]
        if max_step < tot_steps * N:
           pass
        else:
           return i-1

def sample_step_list(step_index=30):
    # The parameter ratio is used for generating the sampling time point.
    qq = 5.0
    pp = qq + 1
    ratio= pp/qq
    # general ratio
    step_list_v2 = [int((ratio)**n) for n in range(step_index)]
    return step_list_v2

