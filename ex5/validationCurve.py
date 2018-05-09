import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def validationCurve(X, y, Xval, yval):
    """returns the train
    and validation errors (in error_train, error_val)
    for different values of lambda. You are given the training set (X,
    y) and validation set (Xval, yval).
    """

# Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.])

# You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in 
#               error_train and the validation errors in error_val. The 
#               vector lambda_vec contains the different lambda parameters 
#               to use for each calculation of the errors, i.e, 
#               error_train(i), and error_val(i) should give 
#               you the errors obtained after training with 
#               lambda = lambda_vec(i)
#
# Note: You can loop over lambda_vec with the following:
#
    count = 0 
    
    for mylambda in lambda_vec:
        fit_theta = trainLinearReg(X, y, mylambda, method='CG', maxiter=5000)
        error_train[count] = linearRegCostFunction(X,y, fit_theta, mylambda)[0]
        error_val[count] = linearRegCostFunction(Xval,yval, fit_theta, mylambda)[0]
        count = count+1

# =========================================================================

    return lambda_vec, error_train, error_val