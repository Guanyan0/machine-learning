import numpy as np
from utils.utils import *
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    ### START CODE HERE ###
    import numpy as np
    # if type(z)=='numpy.ndarray':
    #     g = []
    #     for i in range(len(z)):
    #         g.append(1/(1+np.exp(-z[i])))
    # else:
    #     g = 1/(1+np.exp(-z))
    g = 1 / (1 + np.exp(-z))
    ### END SOLUTION ###

    return g

def compute_cost(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """

    m, n = X.shape

    ### START CODE HERE ###
    total_cost = 0
    epsilon = 1e-15
    for i in range(m):
        f = sigmoid(np.dot(X[i], w)+b)
        f = np.clip(f, epsilon, 1 - epsilon)
        total_cost += -y[i] * np.log(f) - (1 - y[i]) * np.log(1 - f)
    total_cost /= m
    ### END CODE HERE ### 

    return total_cost

X_train, y_train = load_data("data/ex2data1.txt")
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at house_price_prediction w,b: {:.3f}'.format(cost))

