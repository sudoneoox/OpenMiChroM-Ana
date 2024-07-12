#!/bin/bash
if [ $# -eq 0 ]; then
    echo "no arg provided"
    exit
fi

folder=$1
iterations=$2
filename=$3

for ((i=1; i<=$iterations; i++))
do 
    pdb2ndb.py -f "$folder/iteration_$i/$filename.pdb" -n "$folder/iteration_$i/traj_0"
    ndb2cndb.py -f "$folder/iteration_$i/traj_0.ndb" -n "$folder/iteration_$i/traj_0"
done