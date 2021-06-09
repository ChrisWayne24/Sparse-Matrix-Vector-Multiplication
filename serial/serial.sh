#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J serial
#SBATCH -o /home-mscluster/cpillay/proj/serial/serial.out
#SBATCH -e /home-mscluster/cpillay/proj/serial/serial.err

cd $SLURM_SUBMIT_DIR
./q1
