"""
Implement gradient descent for linear regression with multiple features
"""

import numpy as np
from cost_function import load_data, cost_function
from compute_gradient import compute_gradient


def gradient_descent(X_train, y_train, w_init, b_init, alpha, num_iters):
    """
    Perform gradient descent to learn parameters w and b
    
    Args:
        X_train: matrix of features (m, n)
        y_train: vector of targets (m,)
        w_init: initial weights (n,)
        b_init: initial bias (scalar)
        alpha: learning rate
        num_iters: number of iterations
    
    Returns:
        w: learned weights (n,)
        b: learned bias (scalar)
    """

    # Initialize parameters
    w = np.array(w_init, dtype=float)
    b = float(b_init)

    # For each iteration, compute the gradient and update the parameters
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X_train, y_train, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    # Return the learned parameters
    return w, b

def predict_price(x, w, b):
    """
    Predict the price of a house given its features
    
    Args:
        x: vector of features (n,)
        w: learned weights (n,)
        b: learned bias (scalar)
    
    Returns:
        price: predicted price
    """
    return np.dot(w, x) + b


