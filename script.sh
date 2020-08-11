#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o slurm-%j.out



module load Python/3.6.9
source /scratch/sanaawan/PySyft/venv/bin/activate
which python
python examples/tutorials/transfer_learning1.py
