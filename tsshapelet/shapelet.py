from .utils import utils, np
from .barycenters import barycenters
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

    # --------------------------------------------------------------------------------
    # Preprocessing - direct array manipulations
    # --------------------------------------------------------------------------------

    def quantile_normalization(self, quantile = 0.5):
        ''' De-medians the time series '''
        self.series = self.series - np.quantile(self.series, quantile)
        return self
    
    
    def z_normalization(self):
        ''' z = (x-mu)/sigma '''
        mean = np.mean(self.series)
        self.series = (self.series - mean) / np.std(self.series)
        return self
    
    def min_max_normalization(self):
        ''' Normalize the series between zero and one '''
        self.series = (self.series - np.min(self.series)) / (np.max(self.series) - np.min(self.series))
        return self

    def smooth(self, period):
        ''' 
        Performs center moving average smoothing according to the period
        
        Parameters:
            period (int): The number of indices to take the average over.
        '''
        ret = np.cumsum(self.series)
        ret[period:] = ret[period:] - ret[:-period]
        self.series = ret[period - 1:] / period
        return self

    def phase_sync(self, thres = .9):
        '''
        Removes the beginning of the series until the first peak is found.

        Parameters
            thres (float): The threshold to be used to find the first peak.
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
        
        Parameters:
            factor (float): The factor by which to rescale the series. 0.5 interpolates the 
            series to half its original length. In effect, downsampling by half.
        '''
        if (0 > factor > 1):
            raise ValueError('The factor must be between 0 and 1.')

        self.series = utils['interpolate'](self.series, int(len(self.series)*factor))
        return self

    # --------------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------------

    def extract_features(self, **functions):
        '''
        Extracts time series features from the series. Pass in feature extraction
        by keyword pairs. Saves to self.features attribute.
        '''
        for function in functions:
            self.features[function] = functions[function](self.series)

    # --------------------------------------------------------------------------------
    # Candidate extraction
    # --------------------------------------------------------------------------------

    def peak_extraction(self, min_dist = 60, thres = 0.6, max_dist = 150):
        '''
        Extracts the subsequences between the peaks from the series.

        Parameters
            min_dist (int): The minimum distance between peaks.
            thres (float): The threshold to be used to find the peaks.
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
            qty (int): The number of subsequences to be extracted.
            min_dist (int): The minimum length of the subsequences.
            max_dist (int): The maximum length of the subsequences.
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
            window_length (int): The length of the subsequences.
            step (int): The step size between subsequences.
        '''
        self.candidates = []
        for i in np.arange(0, len(self.series) - window_length, step):
            self.candidates.append(self.series[i:i+window_length])
        return self
    
    # --------------------------------------------------------------------------------
    # Shapelet extraction
    # --------------------------------------------------------------------------------

    def random_shapelet(self, qty, min_dist = 60, max_dist = 150, parallel_cores = 1, w = 0.9, metric = 'dtw', verbose = True):
        '''
        Extracts a specified quantity of random shapelet candidates from the dataset, selects the one with the minimum pairwise 
        distance to all others based on a given distance metric, and assigns it as the shapelet for this instance.

        This method randomly selects `qty` shapelet candidates, each with a random length within the specified range (`min_dist`, `max_dist`). 
        It then computes the pairwise distances between all candidates using the specified `metric` and `w` parameters. 
        The candidate with the minimum pairwise distance is chosen as the shapelet for the instance.

        Parameters:
            qty (int): The number of subsequences to be extracted.
            min_dist (int, optional): The minimum length of the subsequences. Defaults to 60.
            max_dist (int, optional): The maximum length of the subsequences. Defaults to 150.
            parallel_cores (int, optional): The number of parallel cores to use for computing pairwise distances. Defaults to 1.
            w (float, optional): The window size parameter for the distance function, used when `metric` is 'dtw'. Defaults to 0.9.
            metric (str, optional): The distance metric to use for computing pairwise distances. Defaults to 'dtw'.
            verbose (bool, optional): If True, prints the progress and results of the extraction and selection process. Defaults to True.

        Note: The effectiveness of the selected shapelet for tasks such as time series classification or clustering depends on the characteristics
        of the dataset and the specified parameters.
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

        if verbose:
            print('Access the random shapelet using the .shapelet attribute')


    def exhaustive_shapelet(self, window_length = 80, step = 1, w = 0.9, metric = 'dtw', parallel_cores = 1, verbose = True):
        '''
        Performs an exhaustive search for the best shapelet within the series by extracting all possible subsequences using a sliding window approach, 
        then selects the shapelet with the minimum pairwise distance based on the specified distance metric.

        Parameters
            window_length (int, optional): The length of the sliding window used to extract subsequences from the series. Defaults to 80.
            step (int, optional): The step size for the sliding window, determining the amount of overlap between consecutive subsequences. Defaults to 1.
            w (float, optional): The window size parameter for the distance function, used when `metric` is 'dtw'. This parameter limits the alignment path in the dynamic time warping calculation. Defaults to 0.9.
            metric (str, optional): The distance metric to use for computing pairwise distances between subsequences. Supports 'dtw' (Dynamic Time Warping) and 'euclidean' (Euclidean distance). Defaults to 'dtw'.
            parallel_cores (int, optional): The number of cores to use for parallel computation of pairwise distances. Defaults to 1.
            verbose (bool, optional): If True, prints informative messages about the progress of shapelet extraction and selection. Defaults to True.

        Note
            The choice of `window_length` and `step` parameters can significantly affect the computational cost and the quality of the extracted shapelet. 
            Smaller steps increase the resolution of the search but require more computation.
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

        if verbose:
            print('Access the exhaustive shapelet using the .shapelet attribute')


    def barycenter_shapelet(self, min_dist = 60, thres = 0.6, max_dist = 150, barycenter = 'interpolated', verbose = True):
        '''
        Extracts shapelet candidates from the series using peak extraction based on specified parameters and then computes a 
        barycenter shapelet from these candidates. The barycenter represents a 'central' shapelet that summarizes the set of candidates.

        Parameters
            min_dist (int, optional): The minimum distance between peaks to be considered separate candidates during peak extraction. Defaults to 60.
            thres (float, optional): The threshold value for peak extraction, used to determine the significance of peaks. Defaults to 0.6.
            max_dist (int, optional): The maximum distance considered for peak extraction. This parameter can limit the search space for peaks. Defaults to 150.
            barycenter (str, optional): The method used to calculate the barycenter from the set of candidates. Supports 'interpolated' or other predefined methods in the `barycenters` dictionary. Defaults to 'interpolated'.
            verbose (bool, optional): If True, prints informative messages about the progress of shapelet extraction and barycenter creation. Defaults to True.

        Note
            This shapelet method is especially effective for cyclical time series data.
        '''
        if verbose:
            print(f'Extracting candidates from the series using peak extraction with a minimum distance of {min_dist} and a threshold of {thres}')

        self.peak_extraction(min_dist, thres, max_dist)

        if verbose:
            print(f'Creating a barycenter from {len(self.candidates)} candidates')

        self.shapelet =  barycenters[barycenter](self.candidates)
        
        if verbose:
            print('Access the barycenter shapelet using the .shapelet attribute')