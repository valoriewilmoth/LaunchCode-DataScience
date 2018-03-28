from computeCostMulti import computeCostMulti
import numpy as np

def gradientDescentMulti(X_norm, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        
        # Gradient step for theta knot:
        xy_diff = (X_norm.dot(theta) - y)
       
        g_step = (alpha/m) * ( np.transpose(X_norm).dot(xy_diff) )
        
        #update theta
        theta = (theta - g_step)        
       
        
        #print([theta , g_step1, g_step0])

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X_norm, y, theta))

    return theta, J_history