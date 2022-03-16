#rm -rf ../data/*

N=128
S=10000
sbatch job_mc.sh 0.08 $N $S
