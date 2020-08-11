#!/bin/bash

#SBATCH -p gpu --gres="gpu:k80:1" 
#SBATCH --time=720:00:00
#SBATCH -o slurm-%j.out
#SBATCH --mem 16GB

# Clear the environment from any previously loaded modules


# Load the module environment suitable for the job
module load Python/3.6.9
source /scratch/sanaawan/PySyft/venv/bin/activate


# And finally run the job​
python ./examples/tutorials/transfer_learning1.py
