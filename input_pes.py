import autograd.numpy as np

# Define number of electronic states
nel = 2
# PES parameters (atomic units)
mass = 2000.0
A = 0.01
B = 1.6
C = 0.005
D = 1.0

# Define diabatic potential matrix elements
# Go nuts with your diabats. Define them however you want.
def V11(x):
    if x > 0.:
        return A*(1 - np.exp(-B*x))
    else:
        return -A*(1 - np.exp(B*x))

def V22(x):
    return -V11(x)

V12 = lambda x: C*np.exp(-D*(x**2))

all_diabats = [V11, V22]
all_couplings = [V12]
