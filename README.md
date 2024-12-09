# Master_thesis

This repository provides a framework for using the **desilike pipeline** to analyze the impact of spectroscopic systematics on clustering and the measurement of cosmological parameters.

**EPFL Master Thesis 2024**  [Thesis Link](https://www.overleaf.com/read/tctqrhxnwnmm#fcc023)
**Title:** *The Impact of Spectroscopic Systematics on the Measurement of Cosmological Parameters with Full-Modeling Algorithm* 

## Requirements
Basic requirements
  - `numpy`
  - `scipy`
  - `matplotlib`

We use the desilike pipeline
  - `cosmoprimo`
  - `desilike`
  - `FOLPS` (theory)
  - `emcee` (sampler)
  - `getdist` (corner plot)

## DATA
The analysis are based on the QUIJOTE simulation (https://quijote-simulations.readthedocs.io/en/latest/)

To add the spectroscopic systematics from to the box data, use 

To compute the pk, we use the pypower (https://github.com/cosmodesi/pypower)

## INSTALLATION
Most scripts are written for **Jupyter Notebooks**. 

To get started, simply clone the repository:
 ```bash
 git clone https://github.com/ShengyuHE/Master_thesis.git










