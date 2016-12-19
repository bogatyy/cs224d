import numpy as np
import random

from q2_sigmoid import sigmoid, sigmoid_grad

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f_and_grad, x):
    """ 
    Gradient check for a function f
    - f_and_grad should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f_and_grad(x) # Evaluate function value at original point

    y = np.copy(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        reldiff = 1.0
        for negative_log_h in xrange(2, 22):
          h = 0.5 ** negative_log_h
          y[ix] = x[ix] + h
          random.setstate(rndstate)
          fy, _ = f_and_grad(y)
          y[ix] = x[ix]
          numgrad = (fy - fx) / h
          if fx != fy:
            reldiff = min(reldiff, abs(numgrad - grad[ix]) / max((1.0, abs(numgrad), abs(grad[ix]))))

        # Compare gradients
        print 'reldiff', reldiff
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad_and_grad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad_and_grad, np.array(123.456))      # scalar test
    gradcheck_naive(quad_and_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad_and_grad, np.random.randn(4,5))   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    sigmoid_and_grad = lambda x: (np.sum(sigmoid(x)), sigmoid_grad(sigmoid(x)))
    gradcheck_naive(sigmoid_and_grad, np.array(1.23456))      # scalar test
    gradcheck_naive(sigmoid_and_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(sigmoid_and_grad, np.random.randn(4,5))   # 2-D test
    gradcheck_naive(sigmoid_and_grad, np.arange(-5.0, 5.0, 0.1))   # range test
    sincos_and_grad = lambda x: (np.sin(x) + np.cos(x), np.cos(x) - np.sin(x))

    gradcheck_naive(sincos_and_grad, np.array(1.0))

    print

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
