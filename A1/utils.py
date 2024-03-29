import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(dim):
    """
    Initialize parameters w and b to zeros.
    
    Arguments:
    dim -- size of the w vector we want (or number of features)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def model_forward(X, w, b):
    """
    Implement the linear regression model.
    
    Arguments:
    X -- data of size (number of features, number of examples)
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    
    Return:
    z -- the linear model's predictions
    """
    z = np.dot(w.T, X) + b
    return z

def predict(X, w, b):
    """
    Predict the output of the linear regression model
    
    Arguments:
    X -- data of size (number of features, number of examples)
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    
    Returns:
    z -- the linear model's predictions
    """
    z = model_forward(X, w, b)
    return z

def compute_cost(z, Y):
    """
    Computes the cost over all examples
    
    Arguments:
    z -- predictions from model
    Y -- true "label" vector
    
    Returns:
    cost -- total cost computed according to the linear regression cost function
    """
    n = Y.shape[1]
    cost = (1/n) * np.sum((z - Y) ** 2)
    return cost

def model_backward(X, Y, z):
    """
    Compute the gradient of the cost function with respect to the parameters.
    
    Arguments:
    X -- input data
    Y -- true "label" vector
    z -- predictions
    
    Returns:
    dw -- gradient of the loss with respect to w
    db -- gradient of the loss with respect to b
    """
    n = X.shape[1]
    dz = (2/n) * (z - Y)
    dw = np.dot(X, dz.T)
    db = np.sum(dz)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    """
    Update parameters using the gradient descent update rule
    
    Arguments:
    w -- weights
    b -- bias
    dw -- gradient of the loss with respect to w
    db -- gradient of the loss with respect to b
    learning_rate -- learning rate
    
    Returns:
    w -- updated weights
    b -- updated bias
    """
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def train_linear_model(X, Y, num_iterations, learning_rate):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    X -- data of shape (number of features, number of examples)
    Y -- true "label" vector of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    
    Returns:
    parameters -- dictionary containing the weights w and bias b
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    dim = X.shape[0]
    w, b = initialize_parameters(dim)
    costs = []
    iterations = []
    
    for i in range(num_iterations):
        
        # Forward propagation and prediction
        z = predict(X, w, b)
        cost = compute_cost(z, Y)
        
        # get the gradients
        dw, db = model_backward(X, Y, z)
        
        w, b = update_parameters(w, b, dw, db, learning_rate)
        
        # Record the costs
        if i % 200 == 0:
            costs.append(cost)
            iterations.append(i)
            print("lr = %f | Cost after iteration %i: %f" %(learning_rate, i, cost))
        if i == num_iterations - 1: # Record the cost for the last iteration
            costs.append(cost)
            iterations.append(num_iterations)
            print("lr = %f | Cost after iteration %i: %f" %(learning_rate, i+1, cost))
    
    parameters = {"w": w, "b": b}
    return parameters, costs, iterations

def plot_costs(costs, iterations, learning_rates, title, show=False):
    """
    Plots the cost function over the iterations for different learning rates.

    Arguments:
    costs -- list of costs for each learning rate
    iterations -- list of iterations
    learning_rates -- list of learning rates
    title -- title of the plot
    """
    
    for lr, cost in zip(learning_rates, costs):
        plt.plot(iterations, cost, label=f'α={lr}')
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.title(title)
    plt.legend()
    filename = title.replace(' ','')
    plt.savefig(f'plots/{filename}.png')
    if show:
        plt.show()
    plt.clf() # Clear the plot