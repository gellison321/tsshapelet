from .utils import utils, np

#-------------------------------------------#
# Helper methods for barycenter computation #
#-------------------------------------------#

def interpolated_average_barycenter(C):
    '''
    Interpolates each time series in a collection of time series to the same length and computes the average time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)

    Returns:
        interpolated_average_barycenter: np.array, shape = (length, )
    '''
    length = np.mean(list(map(len, C)), dtype = int)
    C = np.array([utils['interpolate'](c, length) for c in C])
    return np.mean(C, axis = 0)

def average_barycenter(C):
    '''
    Computes the average time series in a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        
    Returns:
        average_barycenter: np.array, shape = (length, )
    '''
    return np.mean(C, axis = 0)

# def shape_based_barycenter(C):
#     pass
# def dtw_barycenter():
#     pass
# def soft_dtw_barycenter():
#     pass 

barycenters = {'interpolated_average' : interpolated_average_barycenter,
               'average' : average_barycenter,
              }