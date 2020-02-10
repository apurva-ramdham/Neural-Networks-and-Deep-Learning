import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################
    cov = X.T.dot(X)
    eig_val, eig_vec = np.linalg.eig(cov)
    P = (eig_vec[:,np.argsort(eig_val)[-1:-K-1:-1]]).T
    T = eig_val[np.argsort(eig_val)[-1:-K-1:-1]]
    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
