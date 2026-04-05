"""
Compute the cost function for linear regression with multiple features
Isolate X_train (matrix of features) and y_train (vector of targets) from the dataset(data.csv)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Load the dataset
def load_data():
    """
    Load the dataset from data.csv
    
    Returns:
        X_train: matrix of features
        y_train: vector of targets
    """

    dataset_path = Path(__file__).with_name("data.csv")
    df = pd.read_csv(dataset_path)

    feature_cols = ["size_sqft", "num_bedrooms", "num_floors", "age_of_house_in_months"]
    target_col = "price"

    X_train = df[feature_cols].to_numpy()
    y_train = df[target_col].to_numpy()

    return X_train, y_train

    
def cost_function(w, b, X_train, y_train):
    """
    Compute the cost function for linear regression with multiple features
    
    Args:
        w: array of weights
        b: bias
        X_train: matrix of features
        y_train: vector of targets
    
    Returns:
        cost: cost function value
    """
    # Get the number of training examples
    m = X_train.shape[0]

    # Initialize cost
    cost = 0.0

    # For each training example, compute the sum of squared errors
    for i in range(m):
        f_wb = np.dot(w, X_train[i]) + b
        cost += (f_wb - y_train[i])**2
    
    # Return the average of the squared errors
    cost = cost / (2 * m)
    return cost

    
if __name__ == "__main__":
    # Test the cost function
    w = np.array([1, 1, 1, 1])
    b = 0
    cost = cost_function(w, b, X_train, y_train)
    print(f"Cost: {cost}")
