#!/bin/bash --login
#$ -cwd                   # Run job from directory where submitted

# We now recommend loading the modulefile in the jobscript

module load apps/binapps/anaconda3/2021.11
module load tools/env/proxy2

conda create -n main_env python==3.9.7

source activate main_env

conda install -c conda-forge "ray-tune" 

source deactivate