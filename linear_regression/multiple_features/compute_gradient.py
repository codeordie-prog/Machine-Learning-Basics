"""
Compute the partial derivatives of the cost function with respect to w and b
"""

import numpy as np
from cost_function import load_data

# Load the dataset
X_train, y_train = load_data()

def compute_gradient(X_train, y_train, w, b):
    
    # Get the number of training examples
    m, n = X_train.shape

    # Initialize the gradients
    dj_dw = np.zeros(n)
    dj_db = 0

    # For each training example
    for i in range(m):
        # Calculate the predicted value
        f_wb = np.dot(X_train[i], w) + b
        # Calculate the error
        error = f_wb - y_train[i]
        # Calculate the gradient for each feature
        for j in range(n):
            dj_dw[j] += error * X_train[i][j]
        # Calculate the gradient for the bias
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

