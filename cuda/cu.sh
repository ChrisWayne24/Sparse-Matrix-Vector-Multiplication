#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J simple
#SBATCH -o /home-mscluster/cpillay/proj/cuda/cu.out
#SBATCH -e /home-mscluster/cpillay/proj/cuda/cu.err

cd $SLURM_SUBMIT_DIR
./q1 1024 1 2 cavity02.mtx
