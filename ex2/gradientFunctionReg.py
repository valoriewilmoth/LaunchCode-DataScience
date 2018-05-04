from numpy import asfortranarray, squeeze, asarray

from ex2.gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    
    # Gradient Function for all thetas and will update Theta Zero Second  
    grad = (gradientFunction(theta,X,y) + ((Lambda/m)*theta) )
    
    #update theta zero without regularization
    grad[0] = gradientFunction(theta,X,y)[0]
    
    return grad

#grad = gradientFunctionReg(theta, X, y, 3)