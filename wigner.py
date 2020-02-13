import os
import sys
import math
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

hbar = 1.

# Draw samples from a Gaussian distribution!

def wigner_function(q, p, omega):

    '''
    HO Ground state wigner distribution function.
    omega is the HO frequency
    '''

    return (1./np.pi*hbar)*np.exp(-(p**2 + (q*omega)**2)/(hbar*omega))


def sample_position(x0, omega, N):

    '''
    For HO ground state, the position distribution is:

    1./sqrt(2.*pi*sigma**2) * np.exp(-(x-mu)**2/(2.*sigma**2))

    mu = x0 (centre of gaussian, usually x0=0)
    sigma = sqrt(hbar/2.*omega) STANDARD DEVIATION

    N: number of samples to be drawn

    '''

    sigma = np.sqrt(hbar/(2.*omega))
    
    return np.random.normal(x0, sigma, N)


def sample_momentum(p0, omega, N):

    '''
    '''

    sigma = np.sqrt(omega*hbar/2.)

    return np.random.normal(p0, sigma, N)


# MAIN

x0 = -2.0       # centre
p0 = 0.0
omega = 0.1
sigma_p = np.sqrt(omega*hbar/2.)
sigma_x = np.sqrt(hbar/(2.*omega))

sx = sample_position(x0, omega, 1000)
sp = sample_momentum(p0, omega, 1000)
WDF = wigner_function(sx, sp, omega)

fw = open('init_cond', 'w')
data = np.column_stack((sx, sp, WDF))
np.savetxt(fw, data,fmt='%25.15f')
fw.close()


#sys.exit()

count, bins, ignored = plt.hist(sx, 100, density=True)
plt.plot(bins, 1/(sigma_x * np.sqrt(2 * np.pi)) * np.exp( - (bins - x0)**2 / (2 * sigma_x**2) ), linewidth=2, color='r')
plt.show()
