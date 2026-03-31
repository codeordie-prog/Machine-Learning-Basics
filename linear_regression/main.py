"""
Main file for linear regression
Has the main function to run the program

"""
from gradient_descent import gradient_descent
from cost_function import data

def model(w, b, x):
    """
    Calculate the price prediction by the model
    
    Args:
        w: The weight
        b: The bias
        x: The input value
    
    Returns:
        The predicted price
    """
    return w * x + b


def main():

    # Set learning rate and number of iterations
    alpha = 1e-6
    num_iters = 10000

    # Get the training set
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    
    # Train the model with gradient descent
    w, b = gradient_descent(0, 0, x, y, alpha, num_iters)
    
    # Use the model to predict the price of a house with 97 square meters
    predicted_price = model(w, b, 97)
    print(f"Predicted price: {predicted_price}")

if __name__ == "__main__":
    main()
