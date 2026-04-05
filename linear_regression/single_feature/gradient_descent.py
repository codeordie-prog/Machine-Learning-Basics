"""
Implement the gradient descent algorithm for linear regression


"""
from compute_gradient import compute_gradient
from cost_function import (x, y)

def gradient_descent(w_init, b_init, x, y, alpha, num_iters):

    # initialize the parameters
    w = w_init
    b = b_init
    
    # Iterate over number of iterations
    for _ in range(num_iters):
        # Compute gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    
    return w, b







