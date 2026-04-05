"""
Main script to run multiple features linear regression
"""

from gradient_descent import gradient_descent
from cost_function import load_data
import numpy as np


def predict_price(w, b, x):
    """
    Predict the price of a house given its features
    
    Args:
        w: learned weights (n,)
        b: learned bias (scalar)
        x: vector of features (n,)
    
    Returns:
        price: predicted price
    """
    return np.dot(w, x) + b

def main():
    
    X_train, y_train = load_data()
    w_init = np.zeros(X_train.shape[1])
    b_init = 0.0
    alpha = 1e-6
    num_iters = 100000

    # Train the model
    w, b = gradient_descent(X_train, y_train, w_init, b_init, alpha, num_iters)

    # Use one example from the dataset
    x_example = X_train[0]
    y_example = y_train[0]
    y_pred = predict_price(w, b, x_example)
    print(f"Predicted price: {y_pred}, Actual price: {y_example}")

if __name__ == "__main__":
    main()
