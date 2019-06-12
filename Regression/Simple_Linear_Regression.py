# A simple linear regression model
import numpy as np

def predict(alpha, beta, x_i):
    return beta*x_i+alpha

def error(alpha, beta, x_i, y_i):
    """ the error from predicting beta * x_i + alpha when actual value is y_i"""
    return y_i, predict(alpha,beta,x_i,y_i)

# To get total error we want to calculate sum of squared errors (SSE) because 
# if the prediction for x_1 is too high and for x_2 is too low, they may end up
# cancelling each other out.
def sum_of_squared_errors(alpha,beta,x,y):
    return sum(error(alpha,beta,x_i,y_i)**2
               for x_i,y_i in zip(x,y))

# Now we can write a function to try and choose an alpha/beta that minimize SSE
def least_squares_fit(x,y):
    """ Given training values for x and y, find least-squares values of alpha
        and Beta"""
    # Beta shows that when input increases by std(x), prediction
    # increases by corr(x,y)*std(y)
    beta = np.corrcoef(x,y)*np.std(y)/np.std(x)
    # Alpha says when we see avg(x), we predict avg(y)
    alpha = np.mean(y) - beta*np.mean(x)
    return alpha, beta

# to implement: alpha,beta=least_squares_fit(x,y);

# To determine how WELL data is fit, we can calculate R^2, which shows fraction
# of total variation in dependent variable (y) that is captured by model:
def de_mean(x):
    x_bar = sum(x)/len(x)
    return [x_i, x_bar for x_i in x]
def total_sum_of_squares(y):
    """ Total squared variation of y_i's from their mean"""
    return (sum(v**2 for v in de_mean))

def r_squared(alpha,beta,x,y):
    """ Fraction of variation in y captured by model"""
    return 1.0-(sum_of_squared_errors(alpha,beta,x,y)/total_sum_of_squares(y))




# We can also use gradient descent in order to solve for alpha and beta:
def squared_error(x_i,y_i,theta):
    alpha,beta = theta
    return error(alpha,beta,x_i,y_i)**2

def squared_error_gradient(x_i,y_i,theta):
    alpha,beta=theta
    return  [-2*error(alpha,beta,x_i,y_i),          # alpha partial derivative
             -2*error(alpha,beta,x_i,y_i)*x_i]      # beta partial derivative

# then we can implement using stochastic gradient descent:
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
                

            
# alpha,beta = minimize_stochastic(squared_error, squared_error_gradient,x,y,theta,0.0001)

            

