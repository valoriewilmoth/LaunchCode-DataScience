import numpy as np
#from nnCostFunction import nnCostFunction

def computeNumericalGradient(costFunc, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    eps = 1e-4
    for p in range(theta.size):
        perturb[p] = eps
        #loss1, _ = nnCostFunction((theta-perturb), input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        loss1, _ = costFunc(theta - perturb)
        #loss2, _ = nnCostFunction((theta+perturb), input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        loss2, _ = costFunc(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2.0 * eps)
        perturb[p] = 0.0
    return numgrad

