from ex2.sigmoid import sigmoid
from numpy import squeeze, asarray, reshape
import numpy as np

def gradientFunction(theta, X, y):
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
    #check data type
    if type(X)!=np.array:  X = asarray(X)
    if type(y)!=np.array:  y = asarray(y)
    
    
    h = sigmoid(X.dot(theta))
    
    #check to make sure h and y have the same shape.  
    #having issues with rank 1 arrays (n,)
    #if h.shape!=y.shape: h = np.reshape(h,(-1,1))
    if h.shape!=y.shape: y = y[:,0]
      
    grad = (m**-1) * (X.T.dot(h - y))
      
    return grad

#grad = gradientFunction(initial_theta,X.values,y.values)
