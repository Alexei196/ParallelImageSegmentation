#!/bin/bash
# If permission errors: chmod u+x executable.sh

# To run: ./executable.sh

# This script will run timing on the mpikmeans.cpp and export the stdout to files

# Get path to directory
echo 'type in directory path to images'
read dir_path

# Get execution time of the serial program
mkdir output
mpiexec -n 1 ./build/executables/serialkmeans $dir_path > output/serialkmeans.txt
echo "See serialkmeans.txt for the execution time of the serial program"

# Get execution time for the parallel program(s)
processArray=(1 2 4 8 16 32)
for i in "${processArray[@]}"
do
    echo $i
    mpiexec --oversubscribe -n $i ./build/executables/mpikmeans $dir_path
done > output/mpikmeansParallel.txt
echo "See mpikmeansParallel.txt for the execution time of the parallel MPI program"

for i in "${processArray[@]}"
do
    echo $i
    ~/upcxx/bin/upcxx-run -localhost -n $i ./build/executables/upcxxkmeans $dir_path
done > output/upcxxkmeansParallel.txt
echo "See upcxxkmeansParallel.txt for the execution time of the parallel MPI program"


