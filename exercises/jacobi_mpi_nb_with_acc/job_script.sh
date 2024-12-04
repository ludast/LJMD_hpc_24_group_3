#!/bin/bash
##SBATCH --job-name=jacobi_acc
##SBATCH --output jacobi_acc
#SBATCH -N 4                             # Number of nodes
#SBATCH --ntasks-per-node=4              # Run on all CPU
#SBATCH --cpus-per-task=1                # number of process elements
#SBATCH --time=00:05:00                  # Time limit hrs:min:sec
#SBATCH -p boost_usr_prod
#SBATCH --account=ICT24_MHPC
#SBATCH --mem=32GB
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

cd $SLURM_SUBMIT_DIR

export PGI_ACC_TIME=1
module load nvhpc/24.3
module load openmpi/4.1.6--nvhpc--24.3

srun ./build/bin/jacobi
