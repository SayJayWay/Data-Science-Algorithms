def partial_difference_quotient(f,v,i,h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j==i else 0)       # add h to just the ith element of v
         for j, v_j in enumerate(v)]
    return (f(w)-f(v))/h

def estimate_gradient(f,v,h=0.00001):
    return [partial_difference_quotiet(f,v,i,h)
            for i,_ in enumerate(v)]

def safe(f):
    """Checks function to see if it is safe for our step sizes.  Returns infinity for invalid inputs"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

def step(v,direction,step_size):
    """Move step_size in direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v,direction)]

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):
    """Use gradient descent to find theta that minimizes target function"""
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
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)

