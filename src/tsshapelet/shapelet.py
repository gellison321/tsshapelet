from .utils import barycenters, utils, np
from .comparator import pairwise_argmin

class Shapelet:
    
    ''' The TimeSeries class does preprocessing, feature extraction, 
        and shapelet extraction for 1-dimensional time series data.'''

    def __init__ (self, series):
        self.series = np.array(series)
        if self.series.ndim > 1:
            raise ValueError('The series must be one-dimensional.')
        self.shape = self.series.shape
        self.dtype = self.series.dtype
        self.size = self.series.size
        self.original = self.series
        self.features = {}
        self.candidates = []
        self.shapelets = None

    ##########################################################################
    # The following methods manipulate the series directly for preprocessing #
    ##########################################################################

    def quantile_normalization(self, quantile = 0.5):
        '''
        Normalizes the series by subtracting the quantile from the series.
        
        Parameters
        ----------
        quantile : float
            The quantile to be subtracted from the series.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the normalized series.
            
        '''
        self.series = self.series - np.quantile(self.series, quantile)
        return self
    
    def z_normalization(self):
        '''
        Normalizes the series by subtracting the mean and dividing by the standard deviation.

        Returns
        -------
        self : Shapelet
            The Shapelet object with the z-normalized series.
        '''
        mean = np.mean(self.series)
        self.series = (self.series - mean) / np.std(self.series)
        return self
    
    def min_max_normalization(self):
        '''
        Normalizes the series by subtracting the minimum and dividing by the range.

        Returns
        -------
        self : Shapelet
            The Shapelet object with the min-max normalized series.
        '''
        self.series = (self.series - np.min(self.series)) / (np.max(self.series) - np.min(self.series))
        return self

    def smooth(self, period):
        '''
        Smooths the series by applying a moving average.

        Parameters
        ----------
        period : int
            The period of the moving average.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the smoothed series.
        '''
        ret = np.cumsum(self.series)
        ret[period:] = ret[period:] - ret[:-period]
        self.series = ret[period - 1:] / period
        return self

    def phase_sync(self, thres = .9):
        '''
        Removes the beginning of the series until the first peak is found.

        Parameters
        ----------
        thres : float
            The threshold to be used to find the first peak.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the phase synced series.
        '''
        thres = np.quantile(self.series, thres)
        first = [0]
        first_index = 0
        for i in range(1, len(self.series)-1):
            first_diff = self.series[i] - self.series[i-1]
            first.append(first_diff)
            if i > 1:
                second_diff = first_diff - first[i-1]
                if first_diff < 0 and first[i-1] > 0 and second_diff < 0 and self.series[i] > thres:
                    first_index = i
                    break

        self.series = self.series[first_index:]
        return self

    def rescale(self, factor):
        '''
        Rescales the series by interpolating it to a new length.
        
        Parameters
        ----------
        factor : float
            The factor by which to rescale the series.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the rescaled series.
        '''
        if (0 > factor > 1):
            raise ValueError('The factor must be between 0 and 1.')

        self.series = utils['interpolate'](self.series, int(len(self.series)*factor))
        return self
    
    ######################################################################################################
    # The following methods extract features from the series and add them to the self.features attribute #
    ######################################################################################################

    def extract_features(self, **functions):
        '''
        Extracts time series features from the series.
        
        Parameters
        ----------
        func : 
            The features to be extracted in the form of 
                keyword = function.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted time series features.
        '''
        for function in functions:
            self.features[function] = functions[function](self.series)

    ############################################################
    # The following methods extract candidates from the series #
    ############################################################

    def peak_extraction(self, min_dist = 60, thres = 0.6, max_dist = 150):
        '''
        Extracts the subsequences between the peaks from the series.

        Parameters
        ----------
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        
        Returns
        -------
        self : Shapelet

        '''
        peaks = utils['find_peaks'](self.series, min_dist = min_dist, thres = thres)
        self.candidates = []
        start = 0
        for i in peaks:
            candidate = self.series[start:i]
            start = i 
            if min_dist <= len(candidate) <= max_dist:
                self.candidates.append(candidate)
        return self

    def random_extraction(self, qty, min_dist = 60, max_dist = 150):
        '''
        Extracts random subsequences from the series of a random length within a range.
        
        Parameters
        ----------
        qty : int
            The number of subsequences to be extracted.
        min_dist : int
            The minimum length of the subsequences.
        max_dist : int
            The maximum length of the subsequences.
            
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        self.candidates = []
        for _ in range(qty):
            index = np.random.randint(max_dist, len(self.series)-max_dist)
            length = np.random.randint(min_dist, max_dist) if min_dist != max_dist else max_dist
            self.candidates.append(self.series[index-length//2 : index+length//2])
        return self

    def windowed_extraction(self, window_length = 80, step = 1):
        '''
        Extracts subsequences from the series of a fixed length with a fixed step size.
        
        Parameters
        ----------
        window_length : int
            The length of the subsequences.
        step : int
            The step size between subsequences.
            
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        self.candidates = []
        for i in np.arange(0, len(self.series) - window_length, step):
            self.candidates.append(self.series[i:i+window_length])
        return self
    
    ###########################################################
    # The following methods extract shapelets from the series #
    ###########################################################

    def random_shapelet(self, qty, min_dist = 60, max_dist = 150, parallel_cores = 1, w = 0.9, metric = 'dtw', verbose = True):
        '''
        Extracts a random shapelet from the candidates.

        Parameters
        ----------
        args : 
            The arguments to be passed to the distance function.
            min_dist : int
                The minimum distance of random subsequences.
            thres:
            max_dist
        '''
        if verbose:
            print(f'Extracting {qty} random candidates of a random length in the range: ({min_dist}, {max_dist})')

        self.random_extraction(qty, min_dist, max_dist)

        if verbose:
            print(f'Calculating pairwise distances between {qty} candidates')

        index = pairwise_argmin(self.candidates, w = w, metric = metric, parallel_cores = parallel_cores)

        if verbose:
            print(f'Candidate {index} has the minimum pairwise distance')

        self.shapelet = self.candidates[index]

        print('Access the random shapelet using the .shapelet attribute')

    def exhaustive_shapelet(self, window_length = 80, step = 1, w = 0.9, metric = 'dtw', parallel_cores = 1, verbose = True):
        '''
        Extracts a shapelet from the candidates by calculating the pairwise distances between all candidates. Candidates
        are extracted from the series using a sliding window.

        Parameters
        ----------
        window_length : int
            The length of the subsequences.
        step : int
            The step size between subsequences.
        w : int or float
            The warping window.
        metric : str
            The distance metric to be used.
        parallel_cores : int
            The number of cores to be used for parallel processing.
        verbose : bool
            The verbosity of the output.
        '''
        if verbose:
            print(f'Extracting candidates from the series using a sliding window of length {window_length} and step {step}')

        self.windowed_extraction(window_length, step)

        if verbose:
            print(f'Calculating pairwise distances between {len(self.candidates)} candidates')

        index = pairwise_argmin(self.candidates, w = w, metric = metric, parallel_cores = parallel_cores)

        if verbose:
            print(f'Candidate {index} has the minimum pairwise distance')

        self.shapelet = self.candidates[index]

        print('Access the exhaustive shapelet using the .shapelet attribute')


    def barycenter_shapelet(self, min_dist = 60, thres = 0.6, max_dist = 150, barycenter = 'interpolated', verbose = True):
        '''
        Extracts a shapelet from the candidates by creating a barycenter from the candidates. Candidates
        are extracted from the series using peak extraction.

        Parameters
        ----------
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        max_dist : int
            The maximum distance between peaks.
        barycenter : str
            The type of barycenter to be used.
        verbose : bool
            The verbosity of the output.
        '''
        if verbose:
            print(f'Extracting candidates from the series using peak extraction with a minimum distance of {min_dist} and a threshold of {thres}')

        self.peak_extraction(min_dist, thres, max_dist)

        if verbose:
            print(f'Creating a barycenter from {len(self.candidates)} candidates')

        self.shapelet =  barycenters[barycenter](self.candidates)
        
        print('Access the barycenter shapelet using the .shapelet attribute')