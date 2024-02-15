import sys
sys.path.append('../src/tsshapelet')

from tsshapelet import *
import numpy as np

def query_check(q, c):

    for i, arr in enumerate(c):
        if np.array_equal(q, arr):
            return i

def pairwise_argmin_check(c):

    min_distance = np.inf
    min_index = 0
    for i, arr1 in enumerate(c):
        distance = 0
        for j, arr2 in enumerate(c):
            if i != j:
                distance += metrics['dtw'](arr1, arr2)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index

def test_smooth_trivially(series):
    
    shape = Shapelet(series)
    for i in [1, 10]:
        shape.series = shape.original
        shape.smooth(i)
        shape.series = shape.original

def test_quantile_norm(series):

    shape = Shapelet(series)
    shape.quantile_normalization()
    assert np.quantile(shape.series, 0.5) == 0

def test_min_max_norm(series):

    shape = Shapelet(series)
    shape.min_max_normalization()
    assert np.min(shape.series) == 0
    assert np.max(shape.series) == 1

def test_z_norm(series):

    shape = Shapelet(series)
    shape.z_normalization()
    np.isclose(np.mean(shape.series), 0, atol=1e-10)
    np.isclose(np.std(shape.series), 1, atol=1e-10)

def test_phase_sync(series):

    shape = Shapelet(series)
    shape.phase_sync()
    assert len(shape.series) < len(shape.original)

def test_rescale(series):

    shape = Shapelet(series)
    shape.rescale(1)
    assert len(shape.series) == len(shape.original)
    shape.series = shape.original
    shape.rescale(0.5)
    assert len(shape.series) == len(shape.original)//2

def load_csv(filename, delimiter=',', skip_header=0, dtype = object):
    
    try:
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header, dtype=dtype)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():

    # generating some faux data
    c = np.array([np.sin(np.linspace(0, 10, 80))*.5**(-i*.3) for i in np.linspace(10,15,51)])

    # testing pairwise_argmin
    assert pairwise_argmin_check(c) == pairwise_argmin(c)
    assert pairwise_argmin_check(c) == pairwise_argmin(c, parallel_cores = 4)

    # testing query
    q = np.sin(np.linspace(0, 10, 80))*.5**(-12*.3)
    assert query_check(q, c) == query(q, c)
    assert query_check(q, c) == query(q, c, parallel_cores = 4)
    
    q = np.sin(np.linspace(0, 10, 80))*.5**(-10*.3)
    assert query_check(q, c) == query(q, c)
    assert query_check(q, c) == query(q, c, parallel_cores = 4)

    q = np.sin(np.linspace(0, 10, 80))*.5**(-15*.3)
    assert query_check(q, c) == query(q, c)
    assert query_check(q, c) == query(q, c, parallel_cores = 4)

    # testing euclidean_distance
    arr1 = np.sin(np.linspace(0, 10, 80))
    arr2 = np.sin(np.linspace(0, 10, 80))*.5**(-12*.3)
    assert np.isclose(metrics['euclidean'](arr1, arr2), np.linalg.norm(arr1-arr2))
    assert np.isclose(metrics['euclidean'](arr1, arr2, w = 0.5), np.linalg.norm(arr1[::2]-arr2[::2]))
    
    distance = metrics['euclidean'](arr1, arr2)
    r = distance * 0.5
    assert metrics['euclidean'](arr1, arr2, r = r) < distance

    # testing shapelet
    series = np.array(load_csv('../data/sample_data/001_labeled.csv')[1:,1], dtype=float)

    test_smooth_trivially(series)
    test_quantile_norm(series)
    test_min_max_norm(series)
    test_z_norm(series)
    test_phase_sync(series)
    test_rescale(series)

    print('All tests passed!')

if __name__ == '__main__':
    main()