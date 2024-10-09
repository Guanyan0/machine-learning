import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Load the dataset
X_train, X_val, y_val = load_data()


def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    mu = np.array([])
    var = np.array([])
    ### START CODE HERE ###
    for i in range(n):
        col = [row[i] for row in X]
        mean = np.mean(col)
        mu = np.append(mu, mean)
        var = np.append(var, np.var(col))
    ### END CODE HERE ###

    return mu, var


# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)

print("Mean of each feature:", mu)
print("Variance of each feature:", var)

# UNIT TEST
from public_tests import *

estimate_gaussian_test(estimate_gaussian)

# Returns the density of the multivariate normal
# at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)

# Plotting code
visualize_fit(X_train, mu, var)


# UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        ### START CODE HERE ###
        tp, fp, tn, fn = 0, 0, 0, 0
        prec, rec = 0, 0
        for i in range(y_val.shape[0]):
            if p_val[i] < epsilon:
                if y_val[i] == 1:
                    tp += 1
                elif y_val[i] == 0:
                    fp += 1
            else:
                if y_val[i] == 1:
                    fn += 1
                elif y_val[i] == 0:
                    tn += 1
        if tp == 0:
            prec = 0
            rec = 0
            F1 = 0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            F1 = 2 * prec * rec / (prec + rec)
        ### END CODE HERE ###
        # print(tp, prec, rec, 'epsilon:', epsilon, 'F1:', F1)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1


p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

# UNIT TEST
select_threshold_test(select_threshold)
# Find the outliers in the training set
outliers = p < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)

# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()

print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)
# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))
