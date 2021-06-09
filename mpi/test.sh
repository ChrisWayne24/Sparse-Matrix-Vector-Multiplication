#!/bin/bash

#SBATCH -p batch
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -J mpi
#SBATCH -o /home-mscluster/cpillay/proj/mpi/mpi.out
#SBATCH -e /home-mscluster/cpillay/proj/mpi/mpi.err

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR
mpiexec -n 8 ./q1 1 2 cavity02.mtx
