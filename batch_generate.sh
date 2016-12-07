# /usr/bin/env bash
# generate jobs in batch

processes=(1 2 4 8 16) # The number of threads 
inputs=(benchmark/data/freebase_music.txt) 

for p in ${processes[@]}
do
    ./run_rocks_mpi.sh 1 $p $inputs
done
