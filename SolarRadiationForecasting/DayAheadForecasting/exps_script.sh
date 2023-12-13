#!/bin/bash --login
#$ -cwd                   # Run job from directory where submitted

# If running on a GPU, add:
#$ -l v100=1

#$ -pe smp.pe 8            # Number of cores on a single compute node. GPU jobs can
                            # use up to 8 cores per GPU.

# We now recommend loading the modulefile in the jobscript

module load apps/binapps/anaconda3/2021.11
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/apps/binapps/anaconda3/2022.10/lib/

# $NSLOTS is automatically set to the number of cores requested on the pe line.
# Inform some of the python libraries how many cores we can use.
export OMP_NUM_THREADS=$NSLOTS

source activate ray_env

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

# python create_csv.py
echo "Running main script..."
python script.py