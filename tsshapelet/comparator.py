from .metrics import metrics
import multiprocessing, os, numpy as np

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def find_pool_size(parallel_cores):
    '''
    Helper function for finding the right number of CPU cores to 
    implement in parallel processing.
    
    Parameters:
        parallel_cores int: user input as a starting point
    
    Returns:
        pool_size int: the corrected number of CPU core to implement.
    '''
    
    maximum = os.cpu_count()

    if parallel_cores > maximum:
        parallel_cores = max(1, maximum - 1)
        print(f'Using {parallel_cores} CPU cores.')

    elif parallel_cores < 0:
        parallel_cores = 1
        print(f'Parallel cores must be positive integer. Using 1 CPU core.')

    return parallel_cores


# --------------------------------------------------------------------------------
# query()
# --------------------------------------------------------------------------------

def query_worker(args):
    
    q, c, w, metric = args
    return metrics[metric](q, c, w)


def parallel_query(*args):

    q, C, w, parallel_cores, metric = args

    with multiprocessing.Pool(processes = find_pool_size(parallel_cores)) as pool:
        results = pool.map(query_worker, [(q, c, w, metric) for c in C])

    best_so_far = np.inf
    best_index = None
    for i, dist in enumerate(results):
        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index


def sequential_query(*args):

    q, C, w, metric = args
    best_so_far = np.inf
    best_index = None

    for i in range(len(C)):

        dist = metrics[metric](q, C[i], w = w, r = best_so_far)

        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index


def query(q, C, w = 0.9, metric = 'dtw', parallel_cores = 1):
    '''
    Queries a time series database for the closest match to a query time series, using either a dynamic time warping (dtw) or Euclidean distance metric. The search can be performed either sequentially or in parallel, depending on the number of cores specified.

    Parameters:
        q (Sequence[float]): Time series to query, shape = (q_length,).
        C (Sequence[Sequence[float]]): Library of time series, shape = (n_instances, length).
        w (int or float, optional): Window constraint for the distance functions. Defaults to 0.9.
        metric (str, optional): Distance metric for comparison, either 'dtw' or 'euclidean'. Defaults to 'dtw'.
        parallel_cores (int, optional): Number of cores to use for parallel processing. Defaults to 1. If 1, a sequential procedure is implemented.
    
    Returns:
        int: The index of the time series in C that is closest to the query time series q.
    
    Raises:
        ValueError: If `metric` is not 'dtw' or 'euclidean'.
        ValueError: If `w` is negative.
    
    Examples:
        >>> q = [1, 2, 3, 4, 5]
        >>> C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> query(q, C, w=1, metric='euclidean', parallel_cores=2)
        1
    '''
    if parallel_cores > 1:
        return parallel_query(q, C, w, parallel_cores, metric)
    
    else:
        return sequential_query(q, C, w, metric)
    

# --------------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------------
    
def sequential_score(q, C, metric = 'dtw', w = 0.9):
    scores = []
    for c in C:
        scores.append(metrics[metric](q, c, w = w))
    return np.array(scores)


def score_worker(args):
    q, c, metric, w = args
    return metrics[metric](q, c , w = w)


def parallel_score(q, C, metric = 'dtw', w = 0.9, parallel_cores = 1):

    with multiprocessing.Pool(processes=find_pool_size(parallel_cores)) as pool:
        results = pool.map(score_worker, [(q, c, metric, w) for c in C])

    return np.array(results)


def score(q, C, metric = 'dtw', w = 0.9, parallel_cores = 1):
    '''
    Scores a given query against the library, returning the distance between the query and each
    time series in the corresponding index of the library.
    
    Parameters:
        q (Sequence[float]): Time series to query, shape = (q_length,).
        C (Sequence[Sequence[float]]): Library of time series, shape = (n_instances, length).
        w (Union[int, float]): Window constraint for distance functions. Defaults to 0.9.
        metric (str): Distance metric for comparison, either 'dtw' or 'euclidean'. Defaults to 'dtw'.
        parallel_cores (int): The number of cores to use for parallel processing. Defaults to 1. If 1, a sequential procedure is implemented.
    
    Returns:
        Sequence[float]: An array of scores, each representing the distance between the query and a time series in the library.
    
    Raises:
        ValueError: If `parallel_cores` is not a positive integer.

    Examples:
        >>> q = [1, 2, 3, 4, 5]
        >>> C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> score(q, C, metric='euclidean', w=1, parallel_cores=2)
        [2.0, 1.0, 3.0]
    '''
    if parallel_cores == 1:
        return sequential_score(q, C, metric, w)
    
    elif type(parallel_cores) == int and 0 < parallel_cores:
        return parallel_score(q, C, metric, w)
    
    else:
        print('Parallel_cores must be a positive integer.')


# --------------------------------------------------------------------------------
# pairwise_argmin()
# --------------------------------------------------------------------------------

def pairwise_argmin_worker(args):
    return sum(sequential_score(*args))


def parallel_pairwise_argmin(C, parallel_cores = 1, w = 0.9, metric = 'dtw'):

    with multiprocessing.Pool(processes = find_pool_size(parallel_cores)) as pool:
        results = pool.map(pairwise_argmin_worker, [(q, C, metric, w) for q in C])

    return np.argmin(results)


def sequential_pairwise_argmin(C, metric = 'dtw', w = 0.9):

    min_distance = float('inf')
    min_distance_index = None

    for i in range(len(C)):

        total_distance = 0

        for j in range(len(C)):

            if i != j:
                total_distance += metrics[metric](C[i], C[j], w = w)

        if total_distance < min_distance:
            min_distance = total_distance
            min_distance_index = i

    return min_distance_index


def pairwise_argmin(C, parallel_cores = 1, w = 0.9, metric = 'dtw'):
    '''
    Computes the pairwise minimum argument (argmin) for each pair in a collection
    of time series based on a specified distance metric. This function can operate
    in either parallel or sequential mode depending on the number of parallel_cores specified.

    Parameters:
        C (Sequence[Sequence[float]]): A collection (library) of time series, where each
            time series is represented as a sequence (list, tuple, or similar) of floats.
        parallel_cores (int, optional): The number of cores to use for parallel processing.
            Defaults to 1. If greater than 1, the function operates in parallel mode, speeding
            up the computation for large datasets. If 1, the function operates sequentially.
        w (float, optional): Window constraint for the distance function, applicable to certain
            metrics like 'dtw'. It defines the maximum shift allowed when comparing two sequences.
            Defaults to 0.9.
        metric (str, optional): The distance metric to use for comparing time series. Supported
            values include 'dtw' for Dynamic Time Warping and 'euclidean' for the Euclidean distance.
            Defaults to 'dtw'.

    Returns:
        A structure (list, array, etc.) containing the index of the closest time series in C for each
        time series in the collection, based on the specified metric. The exact type of the return
        value depends on the implementation of the sequential or parallel argmin functions.

    Raises:
        ValueError: If `parallel_cores` is not a positive integer.

    Examples:
        >>> C = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> pairwise_argmin(C, parallel_cores=2, w=1, metric='euclidean')
        [1, 2, 0]  # Example output; actual will depend on the implementation of distance calculation.
    '''
    if parallel_cores > 1:
        return parallel_pairwise_argmin(C, parallel_cores = parallel_cores, w = w, metric = metric)
    
    elif parallel_cores == 1:
        return sequential_pairwise_argmin(C, metric = metric, w = w)
    
    else:
        raise ValueError('Parallel cores should be a positive integer.')