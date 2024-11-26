# spinor_cn
Python code for Crank-Nicolson spin-1 GPE solver

so_pair.py - function definitions for ground state solver for spin-1 system (split-operator Crank-Nicolson)
cs_pair.py - input file that calls so_pair.py
vs_traj.py - vortex tracking algorithm for spinor system 
pq_0.slurm - example SLURM script used with so_pair.py and cs_pair.py
