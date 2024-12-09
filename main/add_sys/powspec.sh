#!/bin/bash

#SBATCH -p p4
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH -J pk
#SBATCH --exclusive

POWdir=/home/astro/shhe/TPIVb/codes/powspec
DATA=/srv/astro/projects/cosmo3d/shhe/QUIJOTE/test
OUTPUT=/home/astro/shhe/projectNU/main/data
redshift=0.5
pk=pk

export OMP_NUM_THREADS=8
for k in `seq 0 99`; do
    ${POWdir}/POWSPEC --conf ${POWdir}/powspec.conf --data ${DATA}/Mnu_ppp_${k}_z${redshift}.dat --auto ${OUTPUT}/pk_z${redshift}/pk_test/Mnu_ppp_${k}_z${redshift}_X_RSD.${pk}
done