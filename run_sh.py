import os
import sys
from subprocess import call, check_call

# Load Initial condition data
# X0, P0 = np.loadtxt('init_cond', usecols=(0,1), unpack=True)

# Run Trajectories ina loop for now, paralleize later
N = 20         # Number of runs
x0 = -4.0
v0 = 0.02
name_dir = 'k_40'

# Create TRAJi
os.mkdir(name_dir)
os.chdir(name_dir)
for i in range(N):
    call(['python', '../molecular_dynamics.py', str(x0), str(v0), str(i)]) 
os.chdir('../')
