"""
Implement the compute gradient function for linear regression

- The cost function communicates the size of the error made by our model
- The goal is to minimize the cost function i.e move it towards a local minimum
- To do this we need to adjust the parameters w and b in the direction of the steepest descent hence the name gradient
- The gradient is the slope of the cost function

J(w, b) - cost function
m - number of training examples
x - input feature
y - target variable

gradient descent : repeat until convergence{
w = w - alpha * dj_dw
b = b - alpha * dj_db
}

dj_dw - partial derivative of the cost function with respect to w
dj_db - partial derivative of the cost function with respect to b

we therefore need to implement a function that returns these values

{
    dj_dw = 1/m * sum((w * x_i + b) - y_i) * x_i
    dj_db = 1/m * sum((w * x_i + b) - y_i)
}

"""

def compute_gradient(x, y, w, b):
    """
    computes the gradient for linear regression
    args:
        x (ndarray): Shape (m,) Input data
        y (ndarray): Shape (m,) Target values
        w, b (scalar): Parameters of the model
    
    returns:
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """

    # Get the size of the training set
    m = x.shape[0]

    # initialize the gradients 
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db