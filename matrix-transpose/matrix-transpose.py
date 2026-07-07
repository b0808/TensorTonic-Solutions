import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    a,b = A.shape
    A_T = np.zeros((b,a))

    for i in range(a):
        for j in range(b):
            A_T[j,i] = A[i,j]
    
    
    return A_T
