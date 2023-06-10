# <p align="center"> tsshapelet
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/timeseriespy">
</p>
</div>

## <p align="center"> A timeseries shapelet extraction tool for python.

### **FEATURES**

#### Timeseries Preprocessing
- Moving Average Smoothing
- Quantile Normalization
- Interpolation (downsampling)
- Phase Syncing

#### Shapelet Candidate Extraction Methods
- Peak Extraction using PeakUtils
- Random Extraction
- Random-Normal Extraction
- Stepped Window Extraction

#### Shapelet Creation Methods
- Exhaustive Pairwise Comparison
- Exhaustive Series Comparison
- Barycenter Averaging

#### Distance Metrics
- Dynamic Time Warping
- Cross Correlation
- Euclidean Distance
- FastDTW


### **DEPENDENCIES**
- Numpy
- SciPy
- DTW-Python
- fastdtw
- PeakUtils

##  <p align="center"> IMPLEMENTATION

```python
import pandas as pd
from timeseriespy.ts import TimeSeries
from timeseriespy.functions import manipulations, metrics, barycenters

df = pd.read_csv('./data/sample_data/001_labeled.csv')
univariate_ts_array = df['waist_vm']


''' Example of barycenter averaging with peak extraction '''

ts = TimeSeries(univariate_ts_array, metric = 'dtw')

# Preprocessing
ts.smooth(5)
ts.quantile_normalization(q = 0.5)
ts.phase_sync(mind_dist = 60, thres = 0.9)

# Candidate extraction from series
ts.candidate_extraction(extraction = 'peak', min_dist = 60, thres = 0.8, max_dist = 120)

# Shapelet creation from candidates
ts.shapelet_creation(barycenter = 'interpolated')


''' Example of basic array manipulations '''

arr1 = univariate_ts_array[:1100]
arr2 = univariate_ts_array[1100:2100]

# 'reinterpolate', 'pad', 'moving_average', 'peak_utils', 'interpolate'
manipulations['interpolate'](arr1, len(arr2)) # -> np.array

# 'dtw', 'fdtw', 'correlation', 'euclidean'
metrics['euclidean'](arr1, arr2) # -> float

# 'euclidean', 'interpolated'
barycenters['interpolated']([arr1, arr2])  # -> np.array

```

### [Full Implementation](https://github.com/gellison321/timeseriespy/blob/main/implementation.ipynb)