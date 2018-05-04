from ex2.costFunction import costFunction
import numpy as np

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================
    
    #calculate cost function with the additional regularization term
    J = (costFunction(theta, X, y) + ( Lambda/(2*m) * np.sum(theta[1:]**2)))
    
    return J

#J = costFunctionReg(theta, X, y, Lambda)