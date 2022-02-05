#!/bin/bash
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n  1
#SBATCH -c  64
#SBATCH -J east
#module load /public1/home/sc91981/anaconda3
export PATH=$PATH:/public1/home/sc91981/anaconda3/bin
B=$1 # invers temperature
N=$2 # num of node at each layer
S=$3 # total steps

OUTPUT_LOG='log.log'
date >> $OUTPUT_LOG
python3 /public1/home/sc91981/east_model/N_beta_mp/src/main.py -N $N -S $S -B $B >> $OUTPUT_LOG
#wait
