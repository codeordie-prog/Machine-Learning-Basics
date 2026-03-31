# HOUSE PRICE PREDICTION MODEL
The goal of this project is to build a linear regression model to predict house prices based on their size.
We will build the model from scratch using python.
The following data will be used:
- Size of the house in square feet
- Price of the house in dollars

The following table shows the data:

| Size (sq ft) | Price ($) |
|--------------|-----------|
| 500          | 50000     |
| 1000         | 100000    |
| 1500         | 150000    |
| 2000         | 200000    |
| 2500         | 250000    |
| 3000         | 300000    |
| 3500         | 350000    |
| 4000         | 400000    |
| 4500         | 450000    |
| 5000         | 500000    |

The idea is simple, use the above data to train a linear regression model and then use it to predict the price of a house based on its size.

To do so we will go step by step through the theoretical mathematics behind linear regression and then implement it in python.

# Visualizing the goal

The following diagram shows the steps of building the prediction model:

Training set (x - features, y - target) -------> Learning Algorithm -------> Function f(x) (the model) -------> prediction (y hat)

The result of the learning algorithm is the function f(x) that best fits the data, such that given a new input x, the model can predict the output y hat with the least error.

The goal is to find the best function f(x) that minimizes the difference between the predicted values (y hat) and the actual values (y).

# Linear Function

The linear function is the simplest form of a function that can be used to predict the output y based on the input x.

The formula for the linear function is:
$$f_{w,b}(x) = wx + b$$

Where w is the weight and b is the bias.

# Cost Function (J(w,b))

The cost function is used to measure the performance of the model. It quantifies the error between predicted values and expected values.

The most common cost function for linear regression is the Mean Squared Error (MSE).

Its denotation is J(w,b) where w is the weight and b is the bias.

The formula for the cost function is:
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

J - Cost function , w - Weight , b - Bias , m - Number of training examples , x - Input feature , y - Target value

What is happening above is that, for each training example (i), we calculate the difference between the predicted value and the actual value , square it, and then sum all the squared differences. Finally, we divide the sum by  2m to get the average squared error.
J communicates how accurate the model is - if J is low, it means the predicted values are close to the actual values, and if J is high, it means the predicted values are far from the actual values.

This means we are trying to find the values of w and b that minimize J.

# Gradient Descent

Remember the goal is to find the values of w and b that minimize J, to achieve this we will use an optimization algorithm called Gradient Descent that iteratively adjusts w and b to reduce J.
For each itaration, we update w and b in a direction that reduces J.
The following formulas are used to update w and b:
$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

whereby:
- $\alpha$ is the learning rate
- $\frac{\partial J}{\partial w}$ is the partial derivative of J with respect to w
- $\frac{\partial J}{\partial b}$ is the partial derivative of J with respect to b

# Deriving the partial derivatives

To derive the partial derivatives, we need to use the chain rule of calculus.
$\frac{\partial J}{\partial w}$ since we know the value of J we can rewrite it as:
$$\frac{\partial J}{\partial w} = \frac{\partial}{\partial w} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right)$$

Focus on one term in the summation:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Apply the chain rule:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})^2 = 2(f_{w,b}(x^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})$$

let $u = f_{w,b}(x^{(i)}) - y^{(i)}$ - This is the inner function

The standard chain rule formula is:
$$\frac{d}{dx} (g(x^n)) = n \cdot g(x^{n-1}) \cdot \frac{d}{dx} g(x)$$



Now we can rewrite the equation as:
$$\frac{\partial}{\partial w} u^2 = 2u \cdot \frac{\partial}{\partial w} u$$

Replacing u with $f_{w,b}(x^{(i)}) - y^{(i)}$:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})^2 = 2(f_{w,b}(x^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})$$

Calculating the partial derivative of the inner function:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)}) = \frac{\partial}{\partial w} (wx^{(i)} + b - y^{(i)})$$


Let x, b and y be constants with respect to w.
Apply the constant rule which states that the derivative of a constant is 0, e.g. $\frac{\partial}{\partial w} (b) = 0$ thus
$$\frac{\partial}{\partial w} (wx^{(i)} + b - y^{(i)}) = \frac{\partial}{\partial w} (wx^{(i)}) + \frac{\partial}{\partial w} (b) - \frac{\partial}{\partial w} (y^{(i)})$$
$$= \frac{\partial}{\partial w} (wx^{(i)}) + 0 - 0$$
$$= \frac{\partial}{\partial w} (wx^{(i)})$$

Now we can apply the power rule to $\frac{\partial}{\partial w} (wx^{(i)})$:
$$\frac{\partial}{\partial w} (wx^{(i)}) = x^{(i)}$$

Therefore:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)}) = x^{(i)}$$

we can now substitute this back into the original equation:
$$\frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)})^2 = 2(f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

rewriting the equation:
$$\frac{\partial}{\partial w} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

the above is partial derivative of J with respect to w

Now we need to find the partial derivative of J with respect to b:
$$\frac{\partial J}{\partial b} = \frac{\partial}{\partial b} \left( \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right)$$

Focus on one term in the summation:
$$\frac{\partial}{\partial b} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Apply the chain rule:
$$\frac{\partial}{\partial b} (f_{w,b}(x^{(i)}) - y^{(i)})^2 = 2(f_{w,b}(x^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial b} (f_{w,b}(x^{(i)}) - y^{(i)})$$

Let $v = f_{w,b}(x^{(i)}) - y^{(i)}$ - This is the inner function

The standard chain rule formula is:
$$\frac{d}{dx} (g(x^n)) = n \cdot g(x^{n-1}) \cdot \frac{d}{dx} g(x)$$

Calculating the partial derivative of the inner function:
$$\frac{\partial}{\partial b} (f_{w,b}(x^{(i)}) - y^{(i)}) = \frac{\partial}{\partial b} (wx^{(i)} + b - y^{(i)})$$

Let wx and y be constants with respect to b.

Therefore:
$$\frac{\partial}{\partial b} (wx^{(i)} + b - y^{(i)}) = 1$$

Since the derivative of b with respect to b is 1.

we can now substitute this back into the original equation:
$$\frac{\partial}{\partial b} (f_{w,b}(x^{(i)}) - y^{(i)})^2 = 2(f_{w,b}(x^{(i)}) - y^{(i)}) \cdot 1$$
$$= 2(f_{w,b}(x^{(i)}) - y^{(i)})$$

Add the summation and constant factor back:
$$\frac{\partial J}{\partial b} = \frac{1}{2m} \sum_{i=1}^{m} 2(f_{w,b}(x^{(i)}) - y^{(i)})$$

cancel out the 2:
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$


Since now we have the partial derivatives of J with respect to w and b, we can update the parameters using gradient descent:
Repeat until convergence:
- Update w (weight): $w := w - \alpha \frac{\partial J}{\partial w}$
- Update b (bias): $b := b - \alpha \frac{\partial J}{\partial b}$

we can now replace the partial derivatives with their calculated values:
- Update w (weight): $w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$
- Update b (bias): $b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$
- $\alpha$ is the learning rate

The above updates have to be done simultaneously (not one after the other).
The update is done iteratively until the cost function J converges.
Converging means that the change in J between iterations is very small indicating that the algorithm has found a local minimum.

The above algorithm is what we aim to implement in code.

# Files
- `cost_function.py` - Contains the cost function implementation
- `data.py` - Contains the data loading and preprocessing
- `data.csv` - Contains the data for training
- `compute_gradient.py` - Contains the compute gradient implementation
- `gradient_descent.py` - Contains the gradient descent implementation
- `main.py` - Contains the main function to run the algorithm











