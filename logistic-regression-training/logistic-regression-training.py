import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    
    """
    X = np.array(X)
    y = np.array(y)

    n,D = X.shape
    w = np.zeros((D,))
    b = 0.0

    for i in range(steps):

        p = _sigmoid(np.dot(X,w)+b)


        gradient_w = np.dot(X.T,(p-y))/n
        gradient_b = np.mean(p-y)

        w = w-lr*gradient_w
        b = b-lr*gradient_b

    return (w,b)

    

    
    
    
    
    # Write code here
    pass