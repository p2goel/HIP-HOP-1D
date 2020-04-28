import os
import sys
import numpy as np

path = 'k_40'

nel = 2
# Runs
N = 20

# All lines to skip before the time step you want to analyze
n_skiprows = 1200

# Initialize
pop = np.zeros((nel, N))
time = np.zeros(N)
state = np.zeros(N)
average_pop = np.zeros(nel)
occupation  = np.zeros(nel)

# Read data
for i in range(N):
    data_pop = np.loadtxt(path+'/populations_'+str(i), skiprows=n_skiprows, max_rows=1, unpack=True)
    data_hop = np.loadtxt(path+'/md_data_'+str(i), usecols=(0,6), skiprows=n_skiprows, max_rows=1, unpack=True)
    time[i] = data_hop[0]
    state[i] = data_hop[1]
    for j in range(nel):
        pop[j,i] = data_pop[j+1]

# Count states
unique, counts = np.unique(state, return_counts=True)
dict_state     = dict(zip(unique, counts))

# Statistical Averaging
for i in range(nel):
    average_pop[i] = np.sum(pop[i,:])/N
    if float(i) in dict_state.keys():
        occupation[i]  = dict_state[float(i)]/N
    else:
        occupation[i] = 0.

print('Average Population')
print(average_pop)
print('Occupation')
print(occupation)
print('Good luck with internal consistency!')


