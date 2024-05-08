#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # should this be something else? got this from NERSC docs, iirc


ceci ubc.yml

# I've also been experimenting with just directly pasting the "OMP_NUM_THREADS=1 ..."
# command here instead of using the ceci command, but it doesn't seem to work either.
