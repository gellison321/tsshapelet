from .utils import utils, np

#-------------------------------------------
# Helper methods for barycenter computation 
#-------------------------------------------

def interpolated_average_barycenter(C):
    '''
    Interpolates each time series in a collection of time series to the same length and computes the average time series.
    
    Parameters:
        C (array-like, shape = (n_instances, length)): The set of time sequences for the barycenter computation.

    Returns:
        interpolated_average_barycenter: np.array, shape = (length, )
    '''
    length = np.mean(list(map(len, C)), dtype = int)
    C = np.array([utils['interpolate'](c, length) for c in C])
    return np.mean(C, axis = 0)

def average_barycenter(C):
    return np.mean(C, axis = 0)

barycenters = {'interpolated' : interpolated_average_barycenter,
               'average' : average_barycenter
               }