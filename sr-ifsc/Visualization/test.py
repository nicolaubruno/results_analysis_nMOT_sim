#
# Libraries
#--
import numpy as np
from matplotlib import pyplot as plt
#--

#
# Data
#--
def M(x):
    M = np.ones((2,2))

    M[0][1] = x
    M[0][0] = 2*x

    return M
#--

for x in range(3):
    print(M(x))