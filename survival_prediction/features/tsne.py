import time
import numpy
import tensorflow as tf

from features.pca import *

def pca_preprocess(X, K):
    N, D = X.shape;
    X = X - np.tile(np.mean(X, 0), (N, 1));
    P, _ = pca_naive(X, K)
    return np.dot(X, P.T).real

def p_entropy(X2, sigma):

    P = np.exp(-X2*sigma)
    P = P / np.sum(P)
    H = -np.dot(np.log2(P), P)

    return P,H

def binary_search(X, perplexity):

    tol = 1e-5
    goal = np.log2(perplexity)
    N, D = X.shape
    sigma = np.ones((N,)).astype('float')
    P = np.zeros((N,N))

    # X2: norm2 distance matrix - NxN
    X2 = (-2*np.dot(X, X.T) + np.sum(X**2, axis=1)).T + np.sum(X**2, axis=1)

    # loop over all points
    tic = time.time()
    for i in range(N):

        sigma_max = np.inf
        sigma_min = 0
        maxiter = 50

        sigma_i = sigma[i]

        for t in range(maxiter):
            X2_i = X2[i, np.concatenate((np.r_[0:i], np.r_[i+1:N]))]
            Pi, Hi = p_entropy(X2_i, sigma_i)
            # binary search for a correct sigma_i
            if abs(Hi-goal) < tol:
                break
            else:
                if Hi > goal:
                    sigma_min = sigma_i
                    if sigma_max == np.inf:
                        sigma_i *= 2
                    else:
                        sigma_i = (sigma_i + sigma_max)/2
                else:
                    sigma_max = sigma_i
                    sigma_i = (sigma_i + sigma_min)/2

        # Set Pi
        sigma[i] = sigma_i
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:N]))] = Pi

    #Return P
    print('binary search takes {} seconds.'.format(time.time()-tic))
    print('the mean of 1/sigma is {}.'.format(np.mean(np.sqrt(1/sigma))))
    return P 
    

def tsne(X, low_dim=2, pca_dim=30, perplexity=30.0):
    """
    tSNE

    Inputs:
    - X: (float) an array of shape(N,D)
    - low_dim: (int) dimenional of output data
    - pca_dim: (int) rather than using the raw data, we can apply
                pca preprocessing to reduce the dimension to pca_dim
    - perplexity:

    Returns;
    - Y: (float) an array of shape (N,low_dim)
    """
    if pca_dim != None:
        X = pca_preprocess(X,pca_dim)

    N,D = X.shape

    P = binary_search(X, perplexity)
    P = (P + P.T) / (2*N)
    P = np.maximum(P, 1e-12)
    P *= 4
    Y = np.random.normal(0, 1e-4, (N,low_dim))

    T = 1000
    # training parameters
    momentum = 0.5 # initial momentum
    V = np.zeros_like(Y)
    lr = 100 # initial learning rate
    beta = 0.8
    kappa = 0.2
    gamma = 0.2
    mu = np.ones_like(V)

    tic = time.time()
    for t in range(T):
        
        Y2 = (-2*np.dot(Y, Y.T) + np.sum(Y**2, axis=1)).T + np.sum(Y**2, axis=1)
        Q_numerator = 1/(1 + Y2)
        Q = Q_numerator
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q,1e-12)

        #tic = time.time()
        dY = np.zeros_like(Y)
        for i in range(N):
            dY[i,:] = 4*np.dot((P[i,:]-Q[i,:])*Q_numerator[i,:], Y-Y[i,:])
        #print("loop time: {}".format(time.time()-tic))
        # vectorized dY for tensorflow
        #tic = time.time()
        #dY_inter = (P-Q)*Q_numerator
        #dY_vec = np.dot(dY_inter, Y) - (np.sum(dY_inter, axis=1) * Y.T).T
        #dY = 4*dY_vec
        #print("vec time: {}".format(time.time()-tic))

        #print('is vectorized dY correct? {}'.format(np.allclose(dY, dY_vec)))
        
        # calculate learning rate
        dY_hat = (1-beta)*dY + beta*V
        mu[np.where(dY*dY_hat>0)] = mu[np.where(dY*dY_hat>0)] + kappa
        mu[np.where(dY*dY_hat<0)] = (1-gamma)*mu[np.where(dY*dY_hat<0)]
        #mu[np.where(dY*V==0)] = mu[np.where(dY*V==0)]
        # update
        if t > 250:
            momentum = 0.8
        V = momentum*V + lr*mu*dY
        Y += V

        # stop early exaggeration
        if t == 100:
            P /= 4

        # verbose: report intermediate result
        if (t+1) % 100 == 0:
            cost = np.sum(P * np.log(P / Q));
            print('The {} th loop cost: {}, computation time: {}'.format(t+1, cost, time.time()-tic))
        
    return Y


########################################################
#                      DEDUG CODE                      #
########################################################

def Hbeta(D = np.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print("Computing pairwise distances...")
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		#if i % 500 == 0:
		#	print("Computing P-values for point ", i, " of ", n, "...")

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		#print(np.sqrt(1/beta[i]))
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
	return P;

def tsne_debug(X = np.array([]), no_dims = 2, initial_dims = 30, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print("Error: array X should have type float.")
		return -1;
	if round(no_dims) != no_dims:
		print("Error: number of dimensions should be an integer.")
		return -1;

	# Initialize variables
	if initial_dims != None:
            X = pca_preprocess(X, initial_dims);

	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = np.random.randn(n, no_dims);
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = binary_search(X, perplexity);
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;# early exaggeration
	P = np.maximum(P, 1e-12);

        # Run iterations
	for iter in range(max_iter):
                # Compute pairwise affinities
                sum_Y = np.sum(np.square(Y), 1);
                num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
                num[range(n), range(n)] = 0;
                Q = num / np.sum(num);
                Q = np.maximum(Q, 1e-12);

                # Compute gradient
                PQ = P - Q;
                for i in range(n):
                        dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

                # vectorized dY
                #dY_inter = (P-Q)*num
                #dY_vec = np.dot(dY_inter, Y) - (np.sum(dY_inter, axis=1) * Y.T).T
                #dY_vec *= 4
                #print('is vectorized dY correct? {}'.format(np.allclose(dY, dY_vec)))

                # Perform the update
                if iter < 20:
                        momentum = initial_momentum
                else:
                        momentum = final_momentum
                gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
                gains[gains < min_gain] = min_gain;
                iY = momentum * iY - eta * (gains * dY);
                Y = Y + iY;
                Y = Y - np.tile(np.mean(Y, 0), (n, 1));

                # Compute current value of cost function
                if (iter + 1) % 100 == 0:
                        C = np.sum(P * np.log(P / Q));
                        print("Iteration ", (iter + 1), ": error is ", C)

                # Stop lying about P-values
                if iter == 100:
                        P = P / 4;
        
	
	# Return solution
	return Y;   
    
#######################################################
#                      END OF DEDUG CODE              #
#######################################################
    
