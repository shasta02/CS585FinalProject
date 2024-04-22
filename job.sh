#!/bin/bash -l

#$ -P cs585
#$ -l h_rt=8:00:00
#$ -m ea
#$ -N rmsprop
#$ -j y
#$ -o rmsprop.out
#$ -pe omp 32


### load your environment and run the job

# example:
module load miniconda/23.1.0
conda activate myNewEnv
python rmsprop.py