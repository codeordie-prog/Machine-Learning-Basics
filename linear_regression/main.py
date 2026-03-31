"""
Main file for linear regression
Has the main function to run the program

"""
from gradient_descent import gradient_descent
from cost_function import data

def calculate_price(model, x):
    """
    Calculate the price prediction by the model
    
    Args:
        model: The model parameters (w, b)
        x: The input value
    
    Returns:
        The predicted price
    """
    w, b = model
    return w * x + b

def main():

    # Set learning rate and number of iterations
    alpha = 1e-6
    num_iters = 10000

    # Get the training set
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    
    # Train the model with gradient descent
    model = gradient_descent(0, 0, x, y, alpha, num_iters)
    print(f"Model: w = {model[0]}, b = {model[1]}")
    
    # Use the model to predict the price of a house with 97 square meters
    predicted_price = calculate_price(model, 97)
    print(f"Predicted price: {predicted_price}")

if __name__ == "__main__":
    main()
