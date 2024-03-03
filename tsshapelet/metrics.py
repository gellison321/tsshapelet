import numpy as np, psutil
from numba import njit
from functools import lru_cache

# --------------------------------------------------------------------------------
# Caching memory allocation
# --------------------------------------------------------------------------------

maxsize = int(psutil.virtual_memory().total / 256)

# --------------------------------------------------------------------------------
# Dynamic Time Warping
# --------------------------------------------------------------------------------

@njit
def dtw_matrix(I, J, w = 0.9, r = np.inf):

    r_squared = r**2 # Squaring the best_cost to avoid computing square roots in quadratic time
    n, m  = len(I), len(J)
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


@lru_cache(maxsize=maxsize)
def dtw_cached(I, J, w = 0.9, r = np.inf):
    return dtw_matrix(np.array(I), np.array(J), w = w, r = r)[-1, -1]**0.5

def dtw(I, J, w = 0.9, r = np.inf):
    '''
    Calculates the Dynamic Time Warping (DTW) distance between two sequences.

    DTW measures the similarity between two temporal sequences, which may vary in speed.
    For instance, similarities in walking patterns could be detected, even if one person
    was walking faster than the other. This function uses a cost matrix from the `dtw_matrix`
    function to compute the square root of the final cumulative distance between the sequences,
    providing the DTW distance.

    Parameters:
        I (np.ndarray): First sequence, a one-dimensional array of numerical data.
        J (np.ndarray): Second sequence, a one-dimensional array of numerical data.
        w (float, optional): Window parameter to limit the search space of the DTW algorithm,
            expressed as a fraction of the maximum series length. The actual window size is
            calculated as `int(max(len(I), len(J)) * w)`. Defaults to 0.9.
        r (float, optional): Early abandon threshold to stop computation if the cumulative
            distance exceeds this value, improving efficiency in some cases. Defaults to `np.inf`,
            which disables early abandonment.

    Returns:
        float: The DTW distance between the two input sequences.

    Examples:
        >>> I = [1, 2, 3]
        >>> J = [2, 3, 4]
        >>> d(I, J)
        1.4142135623730951
    '''
    return dtw_cached(tuple(I), tuple(J), w = w, r = r)


# --------------------------------------------------------------------------------
# Euclidean Distance
# --------------------------------------------------------------------------------

def euclidean_distance(I, J, r = np.inf):
    if r < np.inf:
        dist = 0
        for i in range(len(I)):
            dist += (I[i] - J[i])**2
            if dist > r:
                return dist**0.5
    else:
        return np.linalg.norm(I-J) 
    

@lru_cache(maxsize=maxsize)
def ed_cached(I, J, r = np.inf, w = 1):

    I, J = np.array(I), np.array(J)

    if w <= 0.5:
        if type(w) not in [int, float] or 1 < w < 0:
            raise ValueError('w must be a non-negative number between 0 and 1')
        
        step = int(1/w) if w != 1 else 1
        I, J = I[::step], J[::step]

    return euclidean_distance(I, J, r)


def ed(I, J, r = np.inf, w = 1):
    '''
    Calculates the Euclidean distance between two sequences, potentially utilizing caching for efficiency.

    This function acts as a wrapper around a cached Euclidean distance calculation function, `ed_cached`,
    by converting input sequences into tuples (which are hashable and can be used as keys in a cache) and
    then calling `ed_cached` with these tuple inputs along with optional parameters for early abandonment
    and a window size, which in the context of Euclidean distance, is typically unused but provided for
    interface consistency.

    Parameters:
        I (Iterable): First sequence, an iterable of numerical data (e.g., list, array).
        J (Iterable): Second sequence, an iterable of numerical data (e.g., list, array).
        r (float, optional): Early abandonment threshold. If the cumulative distance at any point
            exceeds this value, the computation can be stopped early. Defaults to `np.inf`, which
            disables early abandonment.
        w (int, optional): Window size parameter. This is included for interface consistency with
            other distance functions but is not used in the Euclidean distance calculation. Defaults
            to 1.

    Returns:
        float: The Euclidean distance between the two input sequences. If caching is enabled and
        the distance has been previously computed, the cached value is returned.

    Examples:
        >>> I = [1, 2, 3]
        >>> J = [4, 5, 6]
        >>> ed(I, J)
        5.196152422706632
    '''
    return ed_cached(tuple(I), tuple(J), r, w)


# --------------------------------------------------------------------------------
# Map for dynamic scoping
# --------------------------------------------------------------------------------

metrics  = {'euclidean' : ed,
            'dtw' : dtw
            }