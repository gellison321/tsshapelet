import numpy as np
# from tscomparator import dtw
from tslearn.metrics import dtw
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks

def interpolate(array, length):
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))

def find_first_peak(array, thres = 0.9):
    thres = np.quantile(array, thres)
    first = [0]
    for i in range(1, len(array)-1):
        first_diff = array[i] - array[i-1]
        first.append(first_diff)
        if i > 1:
            second_diff = first_diff - first[i-1]
            if first_diff < 0 and first[i-1] > 0 and second_diff < 0 and array[i] > thres:
                return i

def reinterpolate(array, window_length):
    length = len(array)
    return np.concatenate([np.tile(array, window_length//length),array[:window_length%length]])

def pad(array, length): 
    return np.pad(array, (0,length - len(array)), 'constant')

def center_moving_average(array, period):
    ret = np.cumsum(array)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

def indexes(array, min_dist = 60, thres = 0.9):
    return find_peaks(array, height=np.quantile(array, thres), distance=min_dist)[0]

def cross_correlation(arr1, arr2, method = 'avg'):
    cases = {'avg': np.mean, 'max' : np.max, 'min' : np.min}
    return cases[method](correlate(arr1, arr2))

def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1-arr2)

def dynamic_time_warping(arr1, arr2, w = None):
    return dtw(arr1, arr2, w = None)

def average_barycenter(X):
    return np.mean(X, axis = 0)

def interpolated_average(X):
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)

metrics  = {'euclidean' : euclidean_distance,
            'correlation' : cross_correlation,
            'dtw' : dynamic_time_warping,
            }

barycenters = {'interpolated' : interpolated_average,
               'average' : average_barycenter
               }

manipulations = {'interpolate' : interpolate,
                 'reinterpolate' : reinterpolate,
                 'pad' : pad,
                 'moving_average' : center_moving_average,
                 'peak_utils' : indexes,
                 'first_peak' : find_first_peak,
                 }