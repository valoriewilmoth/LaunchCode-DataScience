import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm, mu, sigma = X,0,0
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #
    
    # get the number of features in X and norm 1 col at a time 
 
    for i in range(X.shape[1]):
        mu_i = np.mean(X[:,i])   #calculate mean for each col
        sigma_i = np.std(X[:,i])  #calculate sigma for each col
        X_norm[:,i] = ((X_norm[:,i] - mu_i) / sigma_i)  #norm data in col
        
        # want to make an array of all values of mu and sigma
        if i == 0: 
            mu = mu_i
            sigma = sigma_i
        else:
            mu = np.append(mu,mu_i)
            sigma = np.append(sigma,sigma_i)
    # ============================================================
    
    return X_norm, mu, sigma
