import numpy as np
from numba import njit

@njit
def dtw_matrix(I, J, w = 0.9, r = np.inf):

    '''
    Constructs the cost matrix used to compute the dynamic time warping distance between two arrays.
    Allows for early adandon condition of the algorithm.
    
    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
        w
            Window parameter used to limit the search space of the algorithm.
            The window is set to int(max([I_length, J_length])*w). 
        r
            Early abandon condition of the algorithm. If the cost of the current path
            exceeds r, the algorithm is terminated.
    
    Returns:
        cum_sum, shape = (I_length, J_length)
    '''

    # Squaring the best_cost to avoid computing square roots in quadratic time
    r_squared = r**2

    n = len(I)
    m = len(J)
    w = int(max([n, m])*w)
    cum_sum = np.ones((n+1, m+1)) * np.inf
    cum_sum[0, 0] = 0

    # Recurrently computing the cost matrix
    for i in range(1, n+1):

        for j in range(max(1, i-w), min(m, i+w)+1):
            cost = (I[i-1] - J[j-1])**2
            cum_sum[i, j] = cost + min(cum_sum[i-1, j], cum_sum[i, j-1], cum_sum[i-1, j-1])

        # Early abandon if the cost of the current path exceeds r
        if cum_sum[i,:].min() > r_squared:
            return cum_sum
        
    return cum_sum

def dtw(I, J, w = 0.9, r = np.inf):

    # We take the square root here only once
    return dtw_matrix(I, J, w = w, r = r)[-1, -1]**0.5


def ed(arr1, arr2, r = np.inf, w = 1):

    if w != 1:
        if type(w) not in [int, float] or 1 < w < 0:
            raise ValueError('w must be a non-negative number between 0 and 1')
        
        step = int(1/w) if w != 1 else 1
        arr1 = arr1[::step]
        arr2 = arr2[::step]

    return euclidean_distance(arr1, arr2, r)


@njit
def euclidean_distance(arr1, arr2, r = np.inf):

    if r < np.inf:
        dist = 0
        for i in range(len(arr1)):
            dist += (arr1[i] - arr2[i])**2
            if dist > r:
                return dist**0.5
    else:
        return np.linalg.norm(arr1-arr2) 

metrics  = {'euclidean' : ed,
            'dtw' : dtw
            }