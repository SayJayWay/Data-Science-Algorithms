# A simple multiple regression model
import numpy as np
import random

def predict(x_y,beta):
    """Assumes first element of each x_i is 1"""
    return np.dot(x_i,beta)

def error(x_i,y_i,beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i,y_i, beta)**2

def squared_error_gradient(x_i,y_i,beta):
    """ Gradient (with respect to beta) corresponding to the ith squared term"""
    return [-2*x_ij*error(x_i,y_i,beta)
            for x_ij in x_i]

def minimize_stochastic(target_fn, gradient_fn,x,y,theta_0,alpha_0=0.01):
    data = zip(x,y)
    theta = theta_0                                 # initial guess
    alpha = alpha_0                                 # initial step size
    min_theta,min_value = None, float('inf')        # minimum so far
    iterations_with_no_improvements = 0

    # if we ever get to 100 iterations with no improvements, stop
    while iterations_with_no_improvements < 100:
        value = sum(target_fn(x_i,y_i,theta) for x_i, y_i in data)

        if value< min_value:
            # if we've found a new minimum, remember it and go back to original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvements = 0
            alpha = alpha_0

        else:
            # otherwise we aren't improving -> shrink step size
            iterations_with_no_improvements += 1
            alpha*= 0.9

        # Iterate through data in random order
            def in_random_order(theta):
            """ Generates elements of data in random order"""
            indexes = [i for i,_ in enumerate(data)]        # creates list of indexes
            random.shuffle(indexes)                         # shuffles them
            for i in indexes:
                yield data[i]                               # return data in that order
                
        # and take a step for each data point
        for x_i,y_i in_random_order(data):
            gradient_i = gradient_fn(x_i,y_i,theta)
            theta = np.subtract(theta,np.multipily(alpha,gradient_i))

    return min_theta

def estimate_beta(x,y):
    beta_initial=[random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error,squared_error_gradient,x,y,beta_initial,0.001)
