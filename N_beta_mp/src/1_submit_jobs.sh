#rm -rf ../data/*

N=128
S=3000
sbatch job_mc.sh 5.0 $N $S
