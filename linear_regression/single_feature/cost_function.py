"""
Implement the cost function for linear regression

Given the following dataset:

House size (sq ft) | Price ($)
-------------------|------------
1                  | 300
2                  | 500



"""

# Load the dataset
import numpy as np 
import pandas as pd


# Create the dataset from csv
data = pd.read_csv("data.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values


"""
Now we need to implement the cost function:
J(w,b) = 1/2m * sum((w*x_i +b) - y_i)^2 for i=0 to m-1


"""

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input data
        y (ndarray): Shape (m,) Target values
        w, b (scalar): Parameters of the model
    
    Returns:
        cost (scalar): The cost function
    """

    # Get size of the dataset
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost += (w * x[i] + b - y[i]) **2
   
    return cost / (2 * m)



