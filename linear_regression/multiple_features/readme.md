# Multiple Features Linear Regression

This project implements multiple features linear regression using gradient descent.
Previously, we implemented single feature linear regression. Now, we extend it to multiple features.

## Goal

The goal of this project is to predict the price of a house given its features. Example features could be:
- Size of the house
- Number of bedrooms
- Number of floors
- Age of the house

To do this, we implement a function (model) that takes in the features and returns the predicted price.

This function is similar to the single feature linear regression model, but it takes in multiple features instead of just one. 
The function is in the form:

$$f_{w,b}(x) = w_0x_0 + w_1x_1 + ... + w_nx_n + b$$

Where $w_0, w_1, ..., w_n$ are the weights and $b$ is the bias, and $x_0, x_1, ..., x_n$ are the features.

## Training set, Model function and Dot Product

Previously, the training set was a 2D array with 2 columns: the feature and the target. Now, the training set is a 2D array with multiple columns: the features and the target.

m = number of training examples
n = number of features
i = index of training example
j = index of feature

This means, for each i, we have n features, hence x^(i) is a vector of n features, similarly w is a vector of n weights.
That in mind, we can rewrite the model function as:
$$f_{w,b}(x) = wx + b$$
however w and x are now vectors, hence we use the dot product instead of multiplication as before thus the following:
$$f_{w,b}(x) = w \cdot x + b$$

## Cost Function

The cost function for multiple features is the same as the single feature case, but we use the dot product instead of multiplication.

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Where $f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$ and $y^{(i)}$ is the target value for the i-th training example.

Its important to note that the dot product returns a scalar value (a single number), making it similar to the single feature case, although now we are dealing with matrices and vectors instead of just scalars.

## Partial Derivatives

To find the optimal values of w and b, we need to find the partial derivatives of the cost function with respect to w and b.
As before, we will use the chain rule to find the partial derivatives. 
This will lead us to these equations:

$$\frac{\partial}{\partial w} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

and

$$\frac{\partial}{\partial b} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$

The full derivation is shown in the README.md file in the single_feature directory.

Lets focus on the partial derivative with respect to w:
Remember, w and x are now vectors, hence we use the dot product instead of multiplication.
Since we know that:
$f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$ and $y^{(i)}$ we can rewrite the partial derivative with respect to w as:
$$\frac{\partial}{\partial w} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (w \cdot x^{(i)} + b - y^{(i)}) \cdot x^{(i)}$$

Similarly, we can rewrite the partial derivative with respect to b as:
$$\frac{\partial}{\partial b} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (w \cdot x^{(i)} + b - y^{(i)})$$

However, Note that there are n features, hence we have n partial derivatives with respect to w, one for each feature. Thus, we can rewrite the partial derivative with respect to w as:
$$\frac{\partial}{\partial w_n} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (w \cdot x^{(i)} + b - y^{(i)}) \cdot x_n^{(i)}$$

where $n$ represents the feature index.

## Gradient Descent

For the given number of iterations, we will update the values of w and b using the partial derivatives we derived above.
We will ensure that we stick to the simultaneous update rule, where we update all values of w and b at the same time, thus maintaining consistency in our calculations.

Therefore, for each iteration given the learning rate $\alpha$, we will update the values of w and b as follows:

repeat until convergence:
    w = w - α * ∇wJ(w,b)
    b = b - α * ∇bJ(w,b)

however we know the values of the partial derivatives, hence we rewrite the update rules as the following equation:
Previously:
$$w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$

now:
$$w_n := w_n - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x_n^{(i)}$$

and for b:
$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$

These updates are done simultaneously for all n features and b, in each iteration.

## Files

`cost_function.py` - contains the cost function for multiple features
`gradient_descent.py` - contains the gradient descent function for multiple features
`compute_gradient.py` - contains the compute gradient function for multiple features
`data.py` - contains the load data function for the multiple features
`data.csv` - contains the data for the multiple features





