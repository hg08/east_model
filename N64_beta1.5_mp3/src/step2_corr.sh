#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -J AutoCorr

N=64
S=2000
srun -n 1 python3 main_corr.py -B 1.5 -N $N -S $S
