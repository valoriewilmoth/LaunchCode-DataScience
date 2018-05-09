import numpy as np
def linearRegCostFunction(myX, myy, mytheta, mylambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = myy.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#

    
    myh = myX.dot(mytheta)
    #flatten h if necessary to match y shape
    if myh.shape != myy.shape:
        myh = myh.flatten()
    mycost = 1./(2.*m)*np.sum((myh-myy)**2)
    regterm = (mylambda/(2.*m))*np.sum(mytheta[1:]**2)
    J = (mycost + regterm)
    
    
    #grad has same shape as myTheta (2x1)
    grad = (1./m)*(myX.T.dot(myh-myy))
    regterm = (mylambda/m)*(mytheta)
    regterm[0] = 0 #don't regulate bias term
    regterm = regterm.flatten() #had to flatten the regterm to add to grad
    grad = (grad + regterm)

# =========================================================================

    return J, grad