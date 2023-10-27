from tsshapelet.util import metrics, barycenters, manipulations, np

class Shapelet:
    '''The Shapelet class takes in a series and allows for the extraction of shapelets from it.'''

    def __init__ (self, series, metric = 'dtw'):
        self.series = np.array(series)
        self.original = self.series
        self.metric = metric

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
            The Shapelet object with the normalized series.
            
        '''
        self.series = (self.series - np.mean(self.series)) / np.std(self.series)
        return self
    
    def min_max_normalization(self):
        '''

        Normalizes the series by subtracting the minimum and dividing by the maximum.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the normalized series.
            
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
        self.series = manipulations['moving_average'](self.series, period)
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
        self.series = self.series[manipulations['first_peak'](self.series, thres = thres):]
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
        self.series = manipulations['interpolate'](self.series, int(len(self.series)*factor))
        return self

    def peak_analysis(self, min_dist = 60, thres = 0.75):
        '''
        Finds the indices of the peaks in the series.
        
        Parameters
        ----------
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
            
        Returns
        -------
        self : Shapelet
            The Shapelet object with the indices of the peaks.
        '''
        self.indices = manipulations['peak_utils'](self.series, thres = thres, min_dist = min_dist)
        return self
    
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
        self.peak_analysis(min_dist = min_dist, thres = thres)
        self.candidates = []
        start = 0
        for i in self.indices: 
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
        for q in range(qty):
            index = np.random.randint(max_dist, len(self.series)-max_dist)
            length = np.random.randint(min_dist, max_dist) if min_dist != max_dist else max_dist
            self.candidates.append(self.series[index-length//2 : index+length//2])
        return self

    def normal_extraction(self, qty = None, min_dist = None, thres = None):
        '''
        Extracts random subsequnces from the series of a length within one standard deviation of the mean length of the peaks.
        
        Parameters
        ----------
        qty : int
            The number of subsequences to be extracted.
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        if min_dist != None and thres != None:
            self.peak_analysis(min_dist = min_dist, thres = thres)
        else:
            self.peak_analysis()
        lengths = np.diff(self.indices)
        mean = np.mean(lengths, dtype = int)
        std = np.std(lengths, dtype = int)
        qty = len(self.series) // mean if qty == None else qty
        self.random_extraction(qty, min_dist = mean - std, max_dist = mean + std)
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
    
    def _compare_array_to_shapelets(self, array):
        '''
        Helper method for enumerating the distance between a given array and each shapelet in the shapelet list.
        
        Parameters
        ----------
        array : array-like
            The array to be compared to the shapelets.
        
        Returns
        -------
        cost : float
            The sum of the distances between the array and each shapelet.
        '''
        cost = 0
        for shapelet in self.shapelets:
            cost += metrics[self.metric](shapelet, array)
        return cost
 
    def _compare_array_to_series(self, array, step = 1):
        '''
        Helper method for enumerating the distance between a given array and each subsequence of the series, with a fixed step size.
        
        Parameters
        ----------
        array : array-like
            The array to be compared to the subsequences.
        step : int
            The step size between subsequences.
        
        Returns
        -------
        cost : float
            The sum of the distances between the array and each subsequence.
        '''
        size = len(array)
        cost = 0
        for i in range(size, len(self.series) - size, step):
            cost += metrics[self.metric](array, self.series[i:i+size])
        return cost
    
    def _compare_array_to_candidates(self, array):
        '''
        Helper method for enumerating the distance between a given array and each candidate in the candidate list.
        
        Parameters
        ----------
        array : array-like
            The array to be compared to the candidates.
        
        Returns
        -------
        cost : float
            The sum of the distances between the array and each candidate.
        '''
        cost = 0
        for candidate in self.candidates:
            cost += metrics[self.metric](candidate, array)
        return cost

    def order_candidates(self, comparison = 'candidates'):
        '''
        Orders the candidates by their distance to the series.
        
        Parameters
        ----------
        comparison : str
            The method to be used to compare the candidates to the series.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the ordered candidates.
        '''
        distances = {}
        comparisons = {'candidates': self._compare_array_to_candidates,
                       'series' : self._compare_array_to_series,
                       'geometric' : self._compare_array_to_shapelets
                      }
        for candidate in self.candidates:
            distances[comparisons[comparison](candidate)] = candidate
        self.candidates = []
        for distance in sorted(distances):
            self.candidates.append(distances[distance])
        return self

    def reinterpolate_shapelets(self, size):
        '''
        Reinterpolates the shapelets to a new length.
        
        Parameters
        ----------
        size : int
            The length of the new shapelets.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the reinterpolated shapelets.
        '''
        shapelets = []
        for shapelet in self.shapelets:
            shapelets.append(manipulations['reinterpolate'](shapelet, size))
        self.shapelets = shapelets  
        return self

    def candidate_extraction(self, extraction = 'peak', min_dist = 60, thres = 0.8, max_dist = 120, sample = 100, window_length = 100, step = 1):
        '''
        Extracts the candidates from the series.
        
        Parameters
        ----------
        extraction : str
            The method to be used to extract the candidates. {'peak', 'normal', 'random', 'windowed'}
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        max_dist : int
            The maximum distance between peaks.
        sample : int
            The number of candidates to be extracted.
        window_length : int
            The length of the subsequences.
        step : int
            The step size between subsequences.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted candidates.
        '''
        if extraction == 'peak':
            self.peak_extraction(min_dist = min_dist, thres = thres, max_dist = max_dist)
        if extraction == 'normal':
            self.normal_extraction(qty = sample)
        if extraction == 'random':
            self.random_extraction(sample, min_dist = min_dist, max_dist = max_dist)
        if extraction == 'windowed':
            self.windowed_extraction(window_length = window_length, step = step)
        return self

    def shapelet_selection(self, barycenter = None, qty = 1,comparison = 'candidates'):
        '''
        Selects the shapelets from the candidates.

        Parameters
        ----------
        barycenter : str
            The method to be used to select the shapelets. {'average', 'interpolated'}
        qty : int
            The number of shapelets to be selected.
        comparison : str
            The method to be used to compare the candidates to the series. {'candidates', 'series', 'geometric'}
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the selected shapelets.
        '''
        if barycenter != None:
            self.shapelets = [barycenters[barycenter](self.candidates)]
        else:
            self.order_candidates(comparison = comparison)
            self.shapelets = self.candidates[:qty]
        return self