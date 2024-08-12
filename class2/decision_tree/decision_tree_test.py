import numpy as np
import matplotlib.pyplot as plt
from public_tests import *

X_train = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])


def compute_entropy(y):
    import numpy as np
    """
    Computes the entropy for 

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    # You need to return the following variables correctly
    entropy = 0.

    ### START CODE HERE ###

    y_len = len(y)
    if y_len == 0:
        return 0
    p1 = sum(y) / y_len
    if p1 == 0 or p1 == 1:
        return 0
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    ### END CODE HERE ###

    return entropy


print("Entropy at root node: ", compute_entropy(y_train))

# UNIT TESTS
compute_entropy_test(compute_entropy)


def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (ndarray):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (ndarray): Indices with feature value == 1
        right_indices (ndarray): Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = []
    right_indices = []

    ### START CODE HERE ###

    for node in node_indices:
        if X[node, feature] == 1:
            left_indices.append(node)
        else:
            right_indices.append(node)
    ### END CODE HERE ###

    return left_indices, right_indices


root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# UNIT TESTS
split_dataset_test(split_dataset)


def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # You need to return the following variables correctly
    information_gain = 0

    ### START CODE HERE ###

    # Weights
    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)
    # Weighted entropy
    weighted_entropy = w_left * compute_entropy(y_left) + w_right * compute_entropy(y_right)
    # Information gain
    # print('node:',compute_entropy(y),'entropy:',weighted_entropy)
    information_gain = compute_entropy(y[node_indices]) - weighted_entropy
    ### END CODE HERE ###

    return information_gain


info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# UNIT TESTS
compute_information_gain_test(compute_information_gain)


# print(X_train.shape[1])

def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1

    ### START CODE HERE ###
    max_gain = 0
    for feature_index in range(num_features):
        gain = compute_information_gain(X, y, node_indices, feature_index)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature_index
    ### END CODE HERE ##

    return best_feature


best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

# UNIT TESTS
get_best_split_test(get_best_split)

tree = []


def build_tree_recursive(X, y, node_indices, node_name, max_depth, current_depth, parentnode_name):
    if current_depth == max_depth:
        print('reach depth')
        return
    best_feature = get_best_split(X, y, node_indices)
    tree.append({'depth': current_depth, 'node_name': node_name, 'parentnode_name': parentnode_name,
                 'best_feature': best_feature,'contain_node':node_indices})
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)

    build_tree_recursive(X, y, left_indices, node_name+'Left', max_depth, current_depth + 1, node_name)
    build_tree_recursive(X, y, right_indices, node_name+'Right', max_depth, current_depth + 1, node_name)

build_tree_recursive(X_train, y_train, root_indices,'Root',3,0,'')
for node in tree:
    print(node)