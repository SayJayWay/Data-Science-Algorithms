import numpy as np

#Generally we want to iterate through our data in a random order
def in_random_order(theta):
    """Generates elements of data in random order"""
    indexes = [i for i,_ in enumerate(data)]    # creates a list of indices
    random.shuffle(indexes)                     # shuffles them
    for i in indexes:
        yield data[i]                           # return data in that order

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x,y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float('inf')
    iterations_with_no_improvements = 0

    # if we ever get to 100 iterations with no improvements - stop
    while iterations_with_no_improvements < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it and go back to original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvements = 0
            alpha=alpha_0
        else:
            # otherwise we aren't improving -> shrink step size
            iterations_with_no_improvements += 1
            alpha*0.9

        # and take a step for each data point
        for x_i, y_i in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, np.multiply(alpha, gradient_i))

    return min_theta

# If we want to maximize a function instead, we can instead minimize its negative (negative gradient):
def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn)
                               x, y, theta_0, alpha_0)
