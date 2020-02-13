import os
import sys
import math
#import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from input_pes import *

xx = np.linspace(-10., 10., 200)

E1 = np.zeros(len(xx))
E2 = np.zeros(len(xx))
v11 = np.zeros(len(xx))
v22 = np.zeros(len(xx))

for i in range(len(xx)):
    x = xx[i]
    vmat = np.zeros((nel, nel))
    vmat[0,0] = V11(x)
    vmat[0,1] = V12(x)
    vmat[1,0] = vmat[0,1]
    vmat[1,1] = V22(x)
    val, vec = np.linalg.eigh(vmat)
    E1[i] = val[0]
    E2[i] = val[1]
    v11[i] = vmat[0,0]
    v22[i] = vmat[1,1]


plt.plot(xx, v11)
plt.plot(xx, v22)
plt.plot(xx, E1)
plt.plot(xx, E2)
plt.show()
