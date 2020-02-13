import os
import sys
import math
#import numpy as np
from autograd import grad
import autograd.numpy as np
import warnings
warnings.filterwarnings("ignore")

from input_pes import *

# Assert length of all diabats is nel
# And length of coupling list is nel*(nel-1)/2
assert len(all_diabats) == nel;
assert len(all_couplings) == (nel*(nel-1))//2

# Get adibatic energies
def get_energies(x):

    '''
    Diagonalize nel x nel PEM at a given geometry to obtain the
    adaiabtic electronic energies (and also eigenvectors). 
    '''
    
    # Initialize
    V_mat = np.zeros((nel, nel))

    # Fill elements
    for a in range(nel):
        for b in range(nel):
            if a == b:
                V_mat[a][a] = all_diabats[a](x)
            elif b > a:
                # Use floor division for pure integers, unlike Python 2. Duh!
                k = (nel*(nel)//2) - (nel-a)*((nel-a))//2 + b - a - 1     
                V_mat[a][b] = all_couplings[k](x)
                V_mat[b][a] = V_mat[a][b]

    # Diagonalize
    E_val, E_vec  = np.linalg.eigh(V_mat)

    return E_val, E_vec


# Get gradients 'analytically'
def makegradmatrix(x):

    '''
    dV/dx
    '''

    # Make grad matrix (evaluate at current x)
    grad_matrix = np.zeros((nel, nel), dtype=np.ndarray)

    # Use automatic differentiation to get gradients of individual Vmn
    for a in range(nel):
        for b in range(nel):
            if a == b:
                grad_matrix[a, b] = grad(all_diabats[a])(x)
            elif a != b and a > b:
                k = (nel*(nel)//2) - (nel-a)*((nel-a))//2 + b - a - 1 
                grad_matrix[a, b] = grad(all_couplings[k])(x)
                grad_matrix[b, a] = grad_matrix[a, b]
            else:
                pass

    return grad_matrix


def get_gradients_and_nadvec(x):

    '''
    '''
    
    # Initialize
    analytical_gradients = np.zeros(nel)

    # Diagonalize potential matrix
    val, vec = get_energies(x)

    # Make grad matrix (evaluate at current x)
    grad_matrix = makegradmatrix(x)

    # Get gradient
    for a in range(nel):
        analytical_gradients[a] = np.dot(vec[:,a].T, np.dot(grad_matrix, vec[:,a]))
    
    # Get nonadiabatic coupling
    nadvec = np.zeros((nel*(nel-1))//2)
    for a in range(nel):
        for b in range(nel):
            if a != b and a > b:
                k = (nel*(nel)//2) - (nel-a)*((nel-a))//2 + b - a - 1 
                # Nonadiabtic coupling between state a and b 
                numerator = np.dot(vec[:,b].T, np.dot(grad_matrix, vec[:,a])) 
                # TODO: Check for degerenacy of state a and state b
                # Give Warning fo HUGE nadvec and quit if tending to infinity.
                # NOTE: How do QM programs deal with it? 
                nadvec[k] = numerator/(val[b] - val[a])

    return val, analytical_gradients, nadvec


def get_gradient_numerical(x, epsilon):

    '''
    Get gradients numerically only using energies. Compare to analytic.
    '''

    # Get energies at displaced geometries
    val_plus, vec_plus   = get_energies(x + epsilon)
    val_minus, vec_minus = get_energies(x - epsilon)

    # Calc gradient from energies
    numerical_gradients = (val_plus - val_minus)/(2.0*epsilon)

    return numerical_gradients


def analytic_two_states(x):

    '''
    For two states, everything can be done analytically, in closed form expressions.
    We will use this to test the code. It's also pleasing to the eye. 
    For algebraic derivation, see supplementary jupyter notebook.
    '''

    # Initialize
    v11 = all_diabats[0](x)
    v22 = all_diabats[1](x)
    v12 = all_couplings[0](x)
    print(v11, v22, v12)
    grad_matrix = makegradmatrix(x)

    # Eigenvalues
    E_1 = ((v11+v22)/2) - math.sqrt(0.25*((v22-v11)**2) + v12**2)
    E_2 = ((v11+v22)/2) + math.sqrt(0.25*((v22-v11)**2) + v12**2)

    energies = np.array([E_1,E_2], dtype='float64')

    # Gradients
    T1 = 0.5*(grad_matrix[0,0] + grad_matrix[1][1])
    T2 = 0.5*(v22-v11)*(grad_matrix[1][1] - grad_matrix[0][0]) + 2.0*v12*grad_matrix[0][1]
    T3 = 0.5/math.sqrt(0.25*((v22-v11)**2) + v12**2)

    G_1 = T1 - T2*T3
    G_2 = T1 + T2*T3

    gradients = np.array([G_1, G_2], dtype='float64')

    # Non-adiabatic coupling vector
    nad_num = ((v11-v22)/v12)*grad_matrix[0][1] +  grad_matrix[1][1] - grad_matrix[0][0]
    uterm   = math.sqrt((v22-v11)**2 + 4.0*(v12**2))
    den_1   = math.sqrt(4*(v12/((v22-v11) + uterm))**2 + 1.)
    den_2   = math.sqrt(4*(v12/((v11-v22) + uterm))**2 + 1.)
    nad_den = den_1*den_2*uterm
    nadvec = nad_num/nad_den

    # Non-adiabatic coupling vector from NEWTON-X documentation
    term_1 = 1./(1. + (2.*v12/(v22 - v11))**2)
    term_2 = (1./(v22-v11))*grad_matrix[0][1] 
    term_3 = (v12/(v22-v11)**2)*(grad_matrix[1][1] - grad_matrix[0][0])
    nadvec_nx = term_1*(term_2 - term_3)

    return energies, gradients, nadvec, nadvec_nx


