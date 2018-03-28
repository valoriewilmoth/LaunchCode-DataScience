import numpy as np

def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = y.size
    J = 0
    
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
 
    #calculate vectorized cost function for multiple vars 
    
    #calc mat mult of X and theta and subt y
    err = (X.dot(theta) - y)
    
    #compute cost fx
    J = (1/(2*m)) * ( np.transpose(err).dot(err) )

# =========================================================================

    return J

