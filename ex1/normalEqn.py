import numpy as np


def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------

    #calculate normal eqn using equation provided 
    step1 = np.linalg.inv((np.transpose(X).dot(X)))
    theta = step1.dot(np.transpose(X)).dot(y)
# -------------------------------------------------------------

    return theta

# ============================================================

