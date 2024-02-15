# Preprocessing
```python
''' Shapelet class preprocessing features '''

from tsshapelet import Shapelet
import numpy as np, pandas as pd, matplotlib.pyplot as plt

data = pd.read_csv('./data/sample_data/001_labeled.csv')
data = data[data['activity'] == 'walk_sidewalk']['waist_vm'][1000:1500]
```
```python
# smoothing using a moving average filter
shape = Shapelet(data)
shape.smooth(10) # define the smoothing period

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(shape.series, color = 'red');
```
<img alt="GitHub" src="./data/resources/smooth.png?raw=true" width = 75%; height = auto>

```python
# centering on the median
shape = Shapelet(data)
shape.quantile_normalization()

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(shape.series, color = 'red');
```
<img alt="GitHub" src="./data/resources/quantile.png?raw=true" width = 75%; height = auto>

```python
# z normalizing
shape = Shapelet(data)
shape.z_normalization()

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(shape.series, color = 'red');
```
<img alt="GitHub" src="./data/resources/z.png?raw=true" width = 75%; height = auto>

```python
# min max scaling

shape = Shapelet(data)
shape.min_max_normalization()

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(shape.series, color = 'red');
```
<img alt="GitHub" src="./data/resources/minmax.png?raw=true" width = 75%; height = auto>

```python
# phase syncing
shape = Shapelet(data)
shape.phase_sync()

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(shape.series, color = 'red');
```
<img alt="GitHub" src="./data/resources/phase.png?raw=true" width = 75%; height = auto>

```python
# rescaling is like downsampling
shape = Shapelet(data)
shape.rescale(0.5)

# upsampling to compare with the original
series = []
for val in shape.series:
    for i in range(2):
        series.append(val)

plt.figure(figsize = (20, 15))
plt.plot(shape.original, color = 'black')
plt.plot(series, color = 'red');
```
<img alt="GitHub" src="./data/resources/rescale.png?raw=true" width = 75%; height = auto>

# Feature Extraction

```python
''' Feature extraction'''
# Define any feature extraction method as a keyword pair
# features = {'name' : func}

# A suite of basic feature extraction provided:
from features import statistical_features, time_series_features

# Loading test data to test the Shapelet class
data = pd.read_csv('./data/sample_data/001_labeled.csv')
data = data[data['activity'] == 'walk_sidewalk']['waist_vm']
shape = Shapelet(data)

# pass in functions as **kwargs
shape.extract_features(**statistical_features, **time_series_features)

# features saved to the .features attribute as a dictionary
print(shape.features)
```
    >> 
    {
    'min': 0.2908900135790158,
    'max': 2.4179487174048995,
    'mean': 1.086885465644344,
    'median': 1.0780171612734188,
    'var': 0.2199825835874377,
    'std': 0.46902300965670934,
    'skewness': 0.20452977627936064,
    'kurtosis': 1.8903984095232342,
    'iqr': 0.8338869717251004,
    'mean_peak_length': 5029.563829787234,
    'zero_crossings': 0,
    'energy': 13961.177794,
    'mean_crossing_rate': 9962
    }

# Shapelet Extraction
```python
# Loading test data to test the Shapelet class
data = pd.read_csv('./data/sample_data/001_labeled.csv')
data = data[data['activity'] == 'walk_sidewalk']['waist_vm'][1000:3000]
shape = Shapelet(data)
```
## Random Shapelet Extraction
This method extracts a random set of candidates from the series, each one of a random length between the defined min and max distance. It then performs an exhaustive pairwise comparison of each candidate, to select the one with the minimum cumulative distance to all other candidates.

```python
shape.random_shapelet(100, # qty of candidates to extract 
                      min_dist = 60, # minimum length of each candidate
                      max_dist = 100, # maximum length of each candidate
                      parallel_cores = 1, # number of CPU cores to implement in processing
                      w = 0.9, # the warping window constraint for DTW
                      metric = 'dtw', # 'dtw' or 'euclidean
                      verbose = True
                      )
```
    >> Extracting 100 random candidates of a random length in the range: (60, 100)
    >> Calculating pairwise distances between 100 candidates
    >> Candidate 88 has the minimum pairwise distance
    >> Access the random shapelet using the .shapelet attribute

```python 
plt.figure(figsize = (20, 15))
plt.title('Random Shapelet Extraction')
for arr in shape.candidates:
    plt.plot(arr, color = 'gray', alpha = .5)

plt.plot(shape.shapelet, color = 'red', lw = 4);
```
<img alt="GitHub" src="./data/resources/random.png?raw=true" width = 75%; height = auto>

## Exhaustive Shapelet Extraction
This method slides a stepped window along the time series to extract candidates. It then performs an exhaustive pairwise comparison of each candidate, to select the one with the minimum cumulative distance to all other candidates.

```python
shape.exhaustive_shapelet(window_length = 100, # length of sliding window
                          step = 10, # step between starting point of each window
                          parallel_cores = 1, # number of CPU cores to implement in processing
                          w = 0.9, # the warping window constraint for DTW
                          metric = 'dtw', # 'dtw' or 'euclidean
                          verbose = True
                          )
```
    >> Extracting candidates from the series using a sliding window of length 100 and step 10
    >> Calculating pairwise distances between 190 candidates
    >> Candidate 11 has the minimum pairwise distance
    >> Access the exhaustive shapelet using the .shapelet attribute

```python
plt.figure(figsize = (20, 15))
plt.title('Exhaustive Shapelet Extraction')
for arr in shape.candidates:
    plt.plot(arr, color = 'gray', alpha = .5)

plt.plot(shape.shapelet, color = 'red', lw = 4);
```
<img alt="GitHub" src="./data/resources/exhaustive.png?raw=true" width = 75%; height = auto>

## Barycenter Shapelet Extraction
This method extracts subsequences between the cyclical peaks in the data. A barycenter is then constructed from the candidate library.

```python
shape.barycenter_shapelet(min_dist = 60, # minimum distance between peaks in the data
                          max_dist = 150, # maximum distance between peaks in the data
                          thres = 0.6, # minimum quantile each peak achieves
                          barycenter = 'interpolated', # 'interpolated' or 'average'
                          verbose = True
                          )
```

    >> Extracting candidates from the series using peak extraction with a minimum distance of 60 and a threshold of 0.6
    >> Creating a barycenter from 19 candidates
    >> Access the barycenter shapelet using the .shapelet attribute

```python
plt.figure(figsize = (20, 15))
plt.title('Barycenter Shapelet Extraction')
for arr in shape.candidates:
    plt.plot(arr, color = 'gray', alpha = .5)

plt.plot(shape.shapelet, color = 'red', lw = 4);
```
<img alt="GitHub" src="./data/resources/barycenter.png?raw=true" width = 75%; height = auto>

# Dynamic Time Warping and DTW Tools

```python
# constructing a dataset
q = np.sin(np.linspace(0, 10, 80))*.5**(-12*.3)
c = np.array([np.sin(np.linspace(0, 10, 80))*.5**(-i*.3) for i in np.linspace(10,15,51)])

plt.figure(figsize = (20, 15))
for arr in c:
    plt.plot(arr, color = 'gray', alpha = .5)
    
plt.plot(q, color = 'red', lw = 4);
```
<img alt="GitHub" src="./data/resources/sample.png?raw=true" width = 75%; height = auto>

## DTW
A DTW implementation with a simple warping constraint and an early abandon condition. If, while constructing the matrix, r is exceeded, the algorithm abandons early and returns the cumulative distance so far. This is useful for argmin functions utilizing DTW, as we will show.

```python
from tsshapelet import dtw

dtw(q, # first time series array
    c[0], # second time series array
    w = 0.9, # warping window constraint - (0,1)
    r = np.inf # early abandon condition
    )
```
    >> (float)

## Query
This function takes advantage of the early abandon condition of DTW, and performs a search, finding the index in a library of time series, given a query. 

This function can also be parallel processed using the multiprocessing library. Parallel processing is only recommended for very large searches, as the overhead is significant. To set the number of parallel cores to the maximum allowable, simply set the number very high, and the function automatically adjusts.

```python
query(q, # the 1d time series in question
      c, # the library of time series (list of arrays or 2d array)
      w = 0.9, # warping window constraint - (0,1)
      parallel_cores = 1, # number of CPU cores to implement in processing
      metric = 'dtw' # 'dtw' or 'euclidean
      )
```
    >> 20

## Pairwise Argmin
Given a library of time series, this function returns the index of the time series with the minimum cumulative distance to all other time series.


Like with the query() function, this function can also be parallel processed using the multiprocessing library. Parallel processing is only recommended for very large searches, as the overhead is significant. To set the number of parallel cores to the maximum allowable, simply set the number very high, and the function automatically adjusts.

```python
pairwise_argmin(c, # the library of time series (list of arrays or 2d array)
                w = 0.9, # warping window constraint - (0,1)
                parallel_cores = 1, # number of CPU cores to implement in processing
                metric = 'dtw' # 'dtw' or 'euclidean
                )
```
    >> 25