#!/bin/bash

#SBATCH -p p5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH -J pkl
#SBATCH --exclusive

module load GCC/12.2.0
module load OpenMPI/4.1.4
module load Python/3.10.8
source /home/astro/shhe/projectNU/nu_env/bin/activate
module load Python/3.10.8

# export PYTHONPATH=${PYTHONPATH}:/home/astro/shhe/.conda/envs/desilike-env/bin/python
# export PYTHONPATH=${PYTHONPATH}:/home/astro/shhe//projectNU/nu_env/bin/python

# export OMP_NUM_THREADS=16
srun --mpi=pmi2 -n 1 python pkcatas.py
# srun --mpi=pmi2 -n 1 python pkcatas.py
