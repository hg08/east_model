#rm -rf ../data/*

N=128
S=10000
sbatch job_mc.sh 1.39 $N $S
