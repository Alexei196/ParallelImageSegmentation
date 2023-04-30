#!/bin/bash
# If permission errors: chmod u+x

# TODO;
# run mpikmeans timing >> exported to a file
# Get execution time of the serial program
mpirun -n 1 ./mpikmeans > mpikmeansSerial.txt
echo "See mpikmeansSerial.txt for the execution time of the serial program"
# Get execution time for the parallel program(s)
processArray=(1 2 4 8 16 32)
for i in "${processArray[@]}"
do
    mpirun -n $i ./mpikmeans
done > mpikmeansParallel.txt
echo "See mpikmeansParallel.txt for the execution time of the parallel MPI program"

# figure out memory usage for mpi

# how to time in upc++
# figure out memory usage for upc++



