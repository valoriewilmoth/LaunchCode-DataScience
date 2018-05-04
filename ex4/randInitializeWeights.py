import numpy as np

def randInitializeWeights(L_in, L_out):
    """randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing
      connections.

      Note that W should be set to a matrix of size(L_out, 1 + L_in) as the column row of W handles the "bias" terms
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first row of W corresponds to the parameters for the bias units
    #

    ep = np.sqrt(6)/np.sqrt(L_in+L_out)
    print('range of random values should be: ',-ep,' : ',ep)
    
    W = np.random.random([L_out,L_in+1]) 
    W = W*2*ep-ep

    print('range of random values is: ',np.min(W),' : ',np.max(W))
# =========================================================================

    return W

