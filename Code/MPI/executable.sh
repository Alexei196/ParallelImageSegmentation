#!/bin/bash
# If permission errors: chmod u+x

# To run: ./executable.sh

# This script will run timing on the mpikmeans.cpp and export the stdout to files

# Get path to directory
echo 'type in directory path to images'
read dir_path

# Get execution time of the serial program
mpiexec -n 1 ./mpikmeans $dir_path > mpikmeansSerial.txt
echo "See mpikmeansSerial.txt for the execution time of the serial program"

# Get execution time for the parallel program(s)
processArray=(1 2 4 8 16 32)
for i in "${processArray[@]}"
do
    mpiexec --oversubscribe -n $i ./mpikmeans $dir_path
done > mpikmeansParallel.txt
echo "See mpikmeansParallel.txt for the execution time of the parallel MPI program"



