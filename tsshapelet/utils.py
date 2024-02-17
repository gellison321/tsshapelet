import numpy as np
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
         

utils = {'interpolate' : interpolate,
         'reinterpolate' : reinterpolate,
         'pad' : pad,
         'find_peaks' : indexes
         }