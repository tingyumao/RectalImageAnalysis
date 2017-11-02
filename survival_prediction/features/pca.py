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
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    V = np.dot(X.T, X)
    
    T, P = np.linalg.eig(V)

    T = T[:K]
    P = P[:,:K].T

    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)

def pca_iter(X, K):
    """
    PCA -- Iterative PCA for high-dimension data

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy array of shape (N, K), showing the score of each
         component vector
    - R: (float) The residual array (N, D)
    """

    ################################################
    #TODO: Implement iterative PCA                 #
    ################################################
    N, D = X.shape
    #T = np.zeros((N,K))
    T = None
    P = np.zeros((K,D))

    J = 100
    eps = 1e-6

    R = X
    for k in range(K):
        tic = time.time()
        lambda_ = 0
        Tk = R[:,k]
        for j in range(J):
            P[k] = np.dot(R.T, Tk)
            P[k] = P[k] / np.linalg.norm(P[k],2)
            Tk = np.dot(R, P[k])
            lambda_new = np.linalg.norm(Tk,2)
            if abs(lambda_new-lambda_) <= eps:
                break
            lambda_new = lambda_
        R = R - np.dot(Tk[:,np.newaxis], P[k][:,np.newaxis].T)
        print('The {} th component takes {}'.format(k, time.time()-tic))
    ################################################
    #              End of your code                #
    ################################################

    return (P,T,R)
