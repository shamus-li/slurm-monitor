#!/bin/bash

#SBATCH --job-name=collect_slurm_stats
#SBATCH --output=/home/wl757/slurm_monitor/logs/collect_slurm_stats.out
#SBATCH --error=/home/wl757/slurm_monitor/logs/collect_slurm_stats.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=monakhova

# this doesn't actually work because sacct is not available on monakhova-compute-01
source /home/wl757/.bashrc
conda activate 6662
python3 /home/wl757/slurm_monitor/collect_slurm_stats.py
sleep 5
exit 0