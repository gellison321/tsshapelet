from .utils import metrics, np
import itertools, multiprocessing, os

###################################################################################
# query() function returns the index of the best matching series in the database. #
###################################################################################

def query_worker(args):
    '''
    Processes a single query against a time series data point.

    Parameters:
    args (tuple): Contains query series, target series, wedge, best distance so far, and window size.

    Returns:
    float: The distance measure between the query and the series.
    '''
    q, C, w, metric = args

    # Available distance metrics
    return metrics[metric](q, C, w=w)

def parallelized_query(*args):
    '''
    Executes a parallelized query over multiple time series data points.

    Parameters:
    args (tuple): Contains the query series, collection of series, wedge, window size, and best distance so far.

    Returns:
    int: The index of the best matching series.
    '''

    q, C, w, pool_size, metric = args

    pool = multiprocessing.Pool(processes=pool_size)
    args = [(q, C[i], w, metric) for i in range(len(C))]
    results = pool.map(query_worker, args)
    pool.close()
    pool.join()

    best_so_far = np.inf
    best_index = None

    for i, dist in enumerate(results):
        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index

def sequential_query(*args):
    '''
    Executes a sequential query over multiple time series data points. Allows for early abandon condition.
    
    Parameters:
        args (tuple): Contains the query series, collection of series, wedge, window size,
        and best distance so far.
        
    Returns:
        int: The index of the best matching series.
    '''

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
    Queries a time series database for the closest match to a query time series.
    
    Parameters:
        q: array-like, shape = (q_length, )
        C: array-like, shape = (n_instances, length)
        w: int or float
        set_length: int or float
        parallel: bool
    
    Returns:
        best_index: int
    '''

    if parallel_cores == 'all':
        parallel_cores = max(1, os.cpu_count() - 1)
        return parallelized_query(q, C, w, parallel_cores)

    elif parallel_cores > 1:
        if parallel_cores > os.cpu_count():
            parallel_cores = os.cpu_count() - 1
            print(f'Warning: Number of parallel cores exceeds number of available cores. \
                  Using {parallel_cores} cores of {parallel_cores + 1} available.')
            
        return parallelized_query(q, C, w, parallel_cores, metric)
    
    else:
        return sequential_query(q, C, w, metric)
    
    
##################################################################################################
# pairwise_argmin() function returns the index of the series with the minimum pairwise distance. #
##################################################################################################

def pairwise_worker(chunk, C, metric = 'dtw', w=0.9):
    '''
    Worker function for the pairwise_argmin_parallel function.

    Parameters:
        chunk (list): A list of indices to process.
        C (array-like): Collection of time series.
        metric (str): The distance metric to use.
        w (int or float, optional): Window parameter for the DTW algorithm.

    Returns:
        float: The minimum distance.
        int: The index of the time series with the minimum pairwise distance.
    '''
    min_distance = float('inf')
    min_distance_index = None
    for i in chunk:
        total_distance = 0
        for j in range(len(C)):
            if i != j:
                total_distance += metrics[metric](C[i], C[j], w=w)
        if total_distance < min_distance:
            min_distance = total_distance
            min_distance_index = i
    return min_distance, min_distance_index

def pairwise_argmin_parallel(C, parallel_cores = 1, w=0.9, metric='dtw'):
    '''
    Computes the pairwise minimum argument in a parallelized manner across multiple time series.

    Parameters:
        C (array-like): Collection of time series.
        parallel_cores (int): The number of parallel cores to use.
        w (int or float, optional): Window parameter for the DTW algorithm.
        metric (str): The distance metric to use.

    Returns:
        int: The index of the time series with the minimum pairwise distance.
    '''

    chunks = [range(i, len(C), parallel_cores) for i in range(parallel_cores)]

    with multiprocessing.Pool(parallel_cores) as pool:
        results = pool.starmap(pairwise_worker, zip(chunks, itertools.repeat(C), itertools.repeat(metric), itertools.repeat(w)))

    min_distance, min_distance_index = min(results, key=lambda x: x[0])
    return min_distance_index


def sequential_pairwise_argmin(C, metric = 'dtw', w = 0.9):
    '''
    Calculates the pairwise minimum argument for a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        parallel: bool
        
    Returns:
        min_distance_index: int
    '''
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
    Calculates the pairwise minimum argument for a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        parallel: bool
        
    Returns:
        min_distance_index: int
    '''
    if type(parallel_cores) != int or parallel_cores < 0:        
        raise ValueError('parallel_cores must be a positive integer.')

    if parallel_cores > 1:

        if parallel_cores > os.cpu_count():

            parallel_cores = os.cpu_count() - 1
            print(f'Using {parallel_cores} of {parallel_cores + 1} available CPU cores.')
            
        return pairwise_argmin_parallel(C, parallel_cores = parallel_cores, w = w, metric = metric)
    
    elif parallel_cores == 1:

        return sequential_pairwise_argmin(C, metric = metric, w = w)