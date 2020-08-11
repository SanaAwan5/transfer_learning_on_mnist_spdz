#!/bin/bash

#SBATCH -p gpu 
#SBATCH --gres="gpu:k80:1" 
#SBATCH --output=slurm-%j.out



module load Python/3.6.9
source /scratch/sanaawan/PySyft/venv/bin/activate
which python
cd examples/tutorials/
python transfer_learning1.py
