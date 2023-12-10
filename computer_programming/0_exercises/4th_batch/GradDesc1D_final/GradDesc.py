import numpy as np

## Gradient descent for the one-dimensional case.
## All the functions here are assumed to have one-dimensional inputs,
## i.e. they are functions R->R. This is true for the functions that
## we wish to optimize and also for their gradients (which are just the
## usual derivatives).

## An auxiliary function that computes the derivative of a given
## function f, evaluated at the point x. It uses finite differences
## to approximate the derivative. The parameter delta controls the
## approximation; it should be small to get a good approximation, but
## not too small, otherwise floating-point approximations dominate.
## The formula is slightly different than the usual one given in an
## analysis course: it uses both x+delta and x-delta to improve the
## accuracy (it's like computing the finite differences on the left
## and on the right of x and taking the average).
def grad(f, x, delta = 1e-5):
    return (f(x + delta) - f(x - delta)) / (2 * delta)

## The generic optimizer. The function f must take a single argument.
## If you have a function of multiple arguments, e.g. g(a, b, c), that
## you want to optimize over one of them, e.g. b, use a lambda, like
## this: `lambda b: g(a, b, c)`
def grad_desc1d(f, x0,            # function and starting point: required
                grad_f = None,     # optionally specify the gradient
                max_t = 100,      # maximum number of iterations
                alpha = 1e-2,     # step (AKA learning rate in machine learning)
                epsilon = 1e-5):  # convergence criterion
    if grad_f is None:
        ## If no gradient was specified, use the grad function
        ## But gradf must be a function of one argument, while
        ## grad takes two arguments: we use a lambda to fix
        ## the first argument (note: this is a closure since
        ## f lives outside of the function)
        grad_f = lambda xx: grad(f, xx)
    x = x0                      # Initialize the value
    xs = [x0]                   # Start collecting the values in a list
    converged = False           # Keep track of convergence
    for k in range(max_t):      # Iterate
        p = grad_f(x)            # Compute the gradient
        x = x - alpha * p       # Update step: move towards the negative gradient
        xs.append(x)            # Store the new value in the list
        if abs(p) < epsilon:    # If the gradient is small enough...
            converged = True    # ...the algorithm converged
            break               # ...no need to continue
    ## Here, we have either converged or ran out of iterations.
    ## Convert the list to an array, it's nicer to work with
    ## once we don't need to change its size any more (e.g. you
    ## can call functions over it and they broadcast)
    xs = np.array(xs)
    ## Done
    return x, xs, converged
