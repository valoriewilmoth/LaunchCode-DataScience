from numpy import round

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#


# =========================================================================

    pred = (sigmoid(X.dot(theta)))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    return pred

#pred = predict(theta,X)