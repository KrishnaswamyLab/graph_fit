import numpy as np

def get_gauss(X, sigma=1):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''
    X_front = X[np.newaxis, :] #Shape [1 x n x p]
    X_back = X[:, np.newaxis] #Shape [n x 1 x p]
    X_differences = X_front - X_back
    D = np.sqrt(np.sum((X_differences**2), axis=-1));
    W = np.exp((-(D**2))/sigma)
    np.fill_diagonal(W,0)
    # return the affinity matrix
    return W