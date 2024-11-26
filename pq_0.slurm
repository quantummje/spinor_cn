#!/bin/bash -l
#SBATCH --job-name=pq_0
#SBATCH --account=a_davis
#SBATCH --partition=general
#SBATCH --time=14-00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH -o out/pq_0.%j.out  # STDOUT
#SBATCH -e out/pq_0.%j.err  # STDERR

python3 cs_pair.py 11 11 0.1 0.00005 1000 -10 160 0.5 0.25 1 0 1 0

