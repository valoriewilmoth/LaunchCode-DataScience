import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0
    
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

    hyp_x = np.dot(X,theta)           #calculate hypothesis
    cf_diff = (hyp_x - y)**2          #calculate diff
    J = (np.sum(cf_diff))/(2*m)       #sum over all data points

# =========================================================================


    return J


