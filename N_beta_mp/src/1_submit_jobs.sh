#rm -rf ../data/*

N=256
S=3000
sbatch job_mc.sh 2.0 $N $S
