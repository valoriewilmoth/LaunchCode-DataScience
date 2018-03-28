from computeCost import computeCost
import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
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
        # Calculate the gradient step according to the equation for theta1:
        g_step1 = (alpha / m * np.sum( (np.dot(X,theta) - y) * X[:,1]) )
        # Gradient step for theta knot:
        g_step0 = (alpha / m * np.sum( (np.dot(X,theta) - y) ) )
        
        #update theta
        theta[0] = (theta[0] - g_step0)
        theta[1] = (theta[1] - g_step1)
        
        #print([theta , g_step1, g_step0])

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
