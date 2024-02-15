import numpy as np
from src.shapelet_dtw import dtw
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def interpolate(array, length):
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))

def reinterpolate(array, window_length):
    length = len(array)
    return np.concatenate([np.tile(array, window_length//length),array[:window_length%length]])

def pad(array, length): 
    return np.pad(array, (0,length - len(array)), 'constant')

def indexes(array, min_dist = 60, thres = 0.9):
    return find_peaks(array, height=np.quantile(array, thres), distance=min_dist)[0]

def euclidean_distance(arr1, arr2, r = np.inf, w = 1):

    if type(w) not in [int, float] or 1 < w < 0:
        raise ValueError('w must be a non-negative number between 0 and 1')
    
    step = int(1/w) if w != 1 else 1

    if w != 1:
        arr1 = arr1[::step]
        arr2 = arr2[::step]
    
    if r < np.inf:
        dist = 0
        for i in range(0,len(arr1), step):
            dist += (arr1[i] - arr2[i])**2
            if dist > r:
                return dist
    else:
        return np.linalg.norm(arr1-arr2)
         

def dynamic_time_warping(arr1, arr2, w = 0.9, r = np.inf):
    return dtw(arr1, arr2, w = w, r = r)

def average_barycenter(X):
    return np.mean(X, axis = 0)

def interpolated_average(X):
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)


metrics  = {'euclidean' : euclidean_distance,
            'dtw' : dynamic_time_warping
            }

barycenters = {'interpolated' : interpolated_average,
               'average' : average_barycenter
               }

utils = {'interpolate' : interpolate,
         'reinterpolate' : reinterpolate,
         'pad' : pad,
         'find_peaks' : indexes
         }