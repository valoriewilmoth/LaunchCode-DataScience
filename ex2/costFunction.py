from numpy import log
from ex2.sigmoid import sigmoid
import numpy as np

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    #make sure X and y are numpy array
    if type(X)!=np.array:  X = np.asarray(X)
    if type(y)!=np.array:  y = np.asarray(y)
    
    #calulcate hyp and cost fx
    h = sigmoid(X.dot(theta))   
    J = (1/m) * ( (-y.T.dot(log(h))) - (1 - y).T.dot(log(1-h)))
    
    return J

#J = costFunction(initial_theta, X, y)
