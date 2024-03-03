from .utils import utils, np

# --------------------------------------------------------------------------------
# Helper methods for statistical feature extraction in the TimeSeries class 
# --------------------------------------------------------------------------------

def skew(array):
    '''
    Computes the skewness of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        skewness: float
    '''
    return np.mean((array - np.mean(array))**3)/(np.std(array)**3)


def kurtosis(array):
    '''
    Computes the kurtosis of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        kurtosis: float
    '''
    return np.mean((array - np.mean(array))**4)/(np.std(array)**4)


def quartile(array):
    '''
    Computes the qth quantile of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        q: float
            Quantile to compute.
    
    Returns:
        quantile: float
    '''
    return np.quantile(array, 0.25)


def iqr(array):
    '''
    Computes the interquartile range of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        iqr: float
    '''
    return np.quantile(array, 0.75) - np.quantile(array, 0.25)


statistical_features = {'min': np.min,
                        'max': np.max,
                        'mean': np.mean,
                        'median': np.median,
                        'var': np.var,
                        'std': np.std,
                        'skewness': skew,
                        'kurtosis': kurtosis,
                        'iqr': iqr
                        }

# --------------------------------------------------------------------------------
# Helper methods for time series feature extraction in the TimeSeries class 
# --------------------------------------------------------------------------------

def mean_peak_length(array):
    '''
    Computes the peak-to-peak distance in an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        peak_to_peak: float
    '''
    peaks_array = utils['find_peaks'](array, thres =  0.9)
    return np.mean([peaks_array[i-1] for i in range(1,len(peaks_array)+1)])

def zero_crossings(array):
    '''
    Computes the number of zero crossings in an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        zero_crossings: int
    '''
    return len(np.where(np.diff(np.sign(array)))[0])

def energy(array):
    '''
    Computes the energy of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
    
    Returns:
        energy: float
    '''
    return np.sum(array**2)


time_series_features = {'mean_peak_length': mean_peak_length,
                        'zero_crossings': zero_crossings,
                        'energy': energy
                        }   