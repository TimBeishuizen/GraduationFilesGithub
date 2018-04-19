import numpy as np


def compute_covariance(X, target_dim = 0):
    """ Computes the covariance of all features on target_dim.

    :param X: The matrix for which covariance is tested.
    :return: A covariance matrix
    """

    if len(X.shape) != 2:
        raise ValueError("The input is not a m by n matrix")
    elif target_dim != 0 and target_dim != 1:
        raise ValueError("The target dimension is not 0 or 1")

    if target_dim == 1:
        X = X.T

    # Compute means
    X = np.float64(X)
    X -= np.mean(X, 1)[:, None]

    #trace_X = np.zeros((X.shape[0], 1))

    #for i in range(X.shape[0]):
    #    trace_X[i, 0] = np.dot(X[i, :], X[i, :])

    cov_X = np.zeros((X.shape[0], X.shape[0]))



    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            comb_mean = np.dot(X[i, :], X[j, :]) / (X.shape[1] - 1)
            # corr_mean = comb_mean / np.sqrt(trace_X[i, 0] * trace_X[j, 0])
            cov_X[i, j] = comb_mean
            cov_X[j, i] = comb_mean
        if i% (int(X.shape[0]/100)) == 0:
            print("Computing covariance... %f done" % (i/X.shape[0]))

    return cov_X
