#!/bin/bash
#SBATCH --job-name=DL
#SBATCH --no-requeue
#SBATCH --nodes=2
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --partition=THIN
#SBATCH --mem=700gb

python3  train_Attention_UNet.py
