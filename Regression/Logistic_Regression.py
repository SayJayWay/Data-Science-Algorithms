# Simple Logistic Regression model
import math
import numpy as np
from functools import reduce, partial

# Logistic Function
def logistic(x):
    return 1.0/(1+math.exp(-x))

# Derivative of logistic function
def logistic_prime(x):
    return logistic(x)*(1-logistic(x))

# Find beta that miximizes log of logistic function
def logistic_log_likelihood_i(x_i, y_i,beta):
    if y_i==1:
        return math.log(logistic(dot(x_i,beta)))
    else:
        return math.log(1-logistic(np.dot(x_i,beta)))

# calculate the product of individual likelihoods of data points
def logistic_log_likelihood(x,y,beta):
    return sum(logistic_log_likelihood_i(x_i,y_i,beta)
               for x_i,y_i in zip(x,y))

# Calculate the gradient
def logistic_log_partial_ij(x_i,y_i, beta,j):
    """ here i is the index of data point, j is index of derivative"""
    return (y_i-logistic(np.dot(x_i,beta)))*x_i[j]

def logistic_log_gradient_i(x_i,y_i,beta):
    """ Gradient of the log likelihood corresponding to 'i'th data point"""
    return [logistic_log_partial_ij(x_i,y_i,beta,j)
            for j,_ in enumerate(beta)]

def logistic_log_gradient(x,y,beta):
    return reduce(vector_add,
                  [logistic_log_gradient_i(x_i,y_i,beta)
                   for x_i,y_i in zip(x,y)])

""" To apply model:
def train_test_split(x,y, test_pct):
    data = zip(x,y)
    train, test = split_data(data, 1-test_pct)
    x_train, y_train = zip(*train)
    x_test,y_test = zip(*test)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y, 0.33)

fn = partial(logistic_log_likelihood, x_train, y_train)
gradient_fn = partial(logistic_log_gradient, x_train, y_train)

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                 # set theta to initial value
    target_fn = safe(target_fn)     # safe version of target_fn
    value = target_fn(theta)        # value we are minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, step_size)
                       for step_size in step_sizes]

        # choose the theta that minimizes error function the most
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # Stop if converging
        if abs(value_next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

# If we want to maximize a function instead, we can instead minimize its negative (negative gradient):
def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)
                          
beta_hat = maximize_batch(fn,gradient_fn,beta_0)

"""

# We can also test the goodness of fit of our model
true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(x_test, y_test):
    predict = logistic(np.dot(beta_hat, x_i))

    if y_i == 1 and predict >= 5:
        true_positives += 1
    elif y_i == 1:
        false_negatives += 1
    elif predict >= 0.5:
        false_positives += 1
    else:
        true_negatives += 1

precision = true_positives/(true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
