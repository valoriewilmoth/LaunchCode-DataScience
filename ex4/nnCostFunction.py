import numpy as np

from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()

    print(Theta1)
    print(Theta2)

# Setup some useful variables
    m, _ = X.shape
    
#add bias layer to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
    #computing cost function without regularization 
    #make sure X and y are numpy array

    
    #calulcate hyp - first do layer 1
    z1 = (X.dot(Theta1.T))
    #add bias term
    z1 = np.column_stack((np.ones((m, 1)), z1))
    a2 = sigmoid(z1)
     
    # Add ones to the a(2) data matrix
    #a2 = np.column_stack((np.ones((m, 1)), a2))
    
    #calculate the hyp for layer 2
    z2 = a2.dot(Theta2.T)
    a3 = sigmoid(z2)
    
    #need to get the y values in a matrix of mx10 0s and 1s
    #create empty array of zeros
    y_vec = np.zeros((m,num_labels))
    #get 1 value for the value
    for row in range(m):
        y_vec[row,y[row]-1] = 1
    
  
    J = (-1/m)*np.sum( (y_vec*np.log(a3)) + (1 - y_vec )*(np.log(1-a3)) )
    
    J = J + Lambda/(2*m) * ( np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2) )
    
    
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
    
    del3 = (a3-y_vec) # m x num_lables (5000 x 10)
    #calculate g prime for the a2, add layer of ones to make shape fit
    
    del2 = np.multiply( del3.dot(Theta2), sigmoidGradient(z1) ) #m x s2 (5000X26)
 
    Theta2_grad = ((1/m)*( a2.T.dot(del3) ).T)
    Theta1_grad = ((1/m)*( X.T.dot(del2) )[:,1:]).T
     
   
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#


    # -------------------------------------------------------------

    Theta1_grad[:,1:] += (Lambda/m) * Theta1[:,1:]
    Theta2_grad[:,1:] += (Lambda/m) * Theta2[:,1:]
    

    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

    # =========================================================================



    return J, grad

#J,grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
