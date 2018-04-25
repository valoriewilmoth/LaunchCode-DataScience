import numpy as np
from scipy.optimize import minimize
from lrCostFunction import lrCostFunction

import sys
sys.path.append('../')
from gradientFunctionReg import gradientFunctionReg


def optimize(Lambda):
    result = minimize(lrCostFunction, initial_theta, method='L-BFGS-B',
               jac=gradientFunctionReg, 
               args=(X.as_matrix(), y, Lambda),
               options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})
    return result



def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    #looping over each number to do the one vs all.
    for num in range(0,num_labels):
        #making create a list of 0 and 1 for the truth for that number
        y_num = y==(num+1)
        y_num = y_num*1  #converting True/False to int
        # Set Initial theta
        initial_theta = np.zeros((n + 1, 1))

        # To test out Cost Function and Gradient Function   
        #J = lrCostFunction(initial_theta, X, y_num, Lambda)
        #grad = gradientFunctionReg(initial_theta, X, y_num, Lambda)
       

        #Minimize using the CostFx and GradFunction 
        result = minimize(lrCostFunction, initial_theta, method='L-BFGS-B',
              jac=gradientFunctionReg, 
              args=(X, y_num, Lambda),
              options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})
        #Put the theta values in the appropriate row
        all_theta[num,:] = result.x
        cost = result.fun
        

        
# =========================================================================

    return all_theta

#Test the program
#a = oneVsAll(X, y, num_labels, Lambda)
