import numpy as np
from numpy import e

def sigmoid(z):
    """computes the sigmoid of z."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).

# =============================================================
    g = 1. / (1. + np.power(e,-z))
    
    return g

#practice values and data types for z
#g = sigmoid(np.array([[100,0],[1,-100]]))
#g = sigmoid(np.array(0))
#print(g)