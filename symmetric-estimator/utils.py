import numpy as np
import pandas as pd
import copy

try:
    import pymbar
except ImportError as e:
    print("pymbar not installed!")
    raise e


def bennett(w_F, w_R):
    """
    Compute free energy difference between states A and B using Bennett Acceptance Ratio
    Parameters:
    w_F: ndarray with shape (NF,)
        works done in forward direction starting from the initial (A) equilibrium ensemble, in unit kT

    w_R: ndarray with shape (NR,)
        works done in forward direction starting from the initial (A) equilibrium ensemble, in unit of kT
    
    Returns:
    df_AB: float 
        free energy difference between states A and B (df_AB = f_B - f_A), in unit of kT
    """

    assert w_F.ndim  == w_R.ndim == 1, "w_F, w_R must be 1d arrays"
    df_AB, ddf = pymbar.BAR(w_F, w_R, relative_tolerance=0.000001, verbose=False, compute_uncertainty=True)

    return df_AB


def time_reversal_of_work(w_t):
    """
    Compute the time-reversed work from the forward work
    Parameters:
    w_t: ndarray with shape (N, T)
        works done in forward direction at each time step, in unit kT
        
    Returns:
    wr_t: ndarray with shape (N, T)
        time-reversed works, in unit kT
    """
    assert w_t.ndim == 2, "w_t must be a 2d array"
    assert np.all(w_t[:, 0] == 0), "all works at t=0 must be zero"

    N = w_t.shape[0]

    dw = w_t[:, 1:] - w_t[:, :-1]
    dw = dw[:, ::-1]                # reverse order
    dw = -dw
    wr_t = np.append(np.zeros([N, 1]), np.cumsum(dw, axis=1), axis=1)
    return wr_t

def time_reversal_of_trajectory(z_t):
    """
    Compute the time-reversed trajectory from the forward trajectory
    Parameters:
    z_t: ndarray with shape (N, T)
        trajectory in forward direction at each time step
    
    Returns:
    zr_t: ndarray with shape (N, T)
        time-reversed trajectory
    """
    assert z_t.ndim == 2, "w_t must be a 2d array"
    return z_t[:, ::-1]

def hist_counts(data, bin_edges, weights):
    """
    Compute the histogram counts of data with weights
    Parameters:
    data: ndarray with shape (N, T)
        data to compute histogram
    bin_edges: ndarray with shape (nbins+1,)
        bin edges
    weights: ndarray with shape (N, T)
        weights of data

    Returns:
    histograms: ndarray with shape (T, nbins)
        histograms at each time step
    """
    assert data.ndim == 2, "data must be 2d array"
    assert weights.ndim == 2, "weights must be 2d array"
    assert bin_edges.ndim == 1, "bin_edges must be 1d array"
    assert data.shape == weights.shape, "data and weights must have the same shape"

    times = data.shape[1]
    nbins = bin_edges.shape[0] - 1

    histograms = np.zeros([times, nbins], dtype=float)
    for time in range(times):
        histograms[time, :], e = np.histogram(data[:, time], bins=bin_edges, weights=weights[:, time], density=False)

    return histograms

def bin_centers(bin_edges):
    """
    Compute the bin centers from bin edges
    Parameters:
    bin_edges: ndarray with shape (nbins+1,)
        bin edges
    
    Returns:
    bin_centers: ndarray with shape (nbins,)
        bin centers """
    return (bin_edges[:-1] + bin_edges[1:]) / 2.

def center_reflection(value, center):
    """
    Reflect the value with respect to the center
    Parameters:
    value: float
        value to reflect
    center: float
        center of reflection

    Returns:
    reflected_value: float
        reflected value
    """
    return -value + 2 * center

def get_bin_indexes(data, bin_edges):
    """
    Get the bin indexes of data given bin edges
    Parameters:
    data: ndarray with shape (N, T)
        data to compute bin indexes
    bin_edges: ndarray with shape (nbins+1,)
        bin edges
    
    Returns:
    bin_indexes: ndarray with shape (N, T)
        bin indexes of data
    """
    assert isinstance(bin_edges, np.ndarray), "bin_edges must be a ndarray"

    cats, _bin_edges = pd.cut(data.ravel(), bin_edges, retbins=True)
    bin_indexes = cats.codes.reshape(data.shape)

    return bin_indexes, _bin_edges


def equal_spaced_bins(list_of_data, nbins, symmetric_center=None):
    """
    Compute equal spaced bins for the data

    Parameters:
    list_of_data: list
        list of data to compute bin edges
    nbins: int
        number of bins  
    symmetric_center: float or None
        center of the symmetric bins, if None, the bins are not symmetric

    Returns:
    bin_edges: ndarray with shape (nbins+1,)
        bin edges
    
    """
    assert isinstance(list_of_data, list), "list_of_data must be a list"
    if symmetric_center is not None:
        assert nbins % 2 == 0, "When symmetric_center is not None, nbins must be even"
        # which mean that the symmetric center is the bin edge right in the middle

    mins = []
    maxs = []
    stds = []

    for data in list_of_data:
        load_data = data[:]
        mins.append(load_data.min())
        maxs.append(load_data.max())
        stds.append(load_data.std())

    min_x = np.min(mins)
    max_x = np.max(maxs)
    std_x = np.min(stds)

    lower = min_x - 0.0000001 * std_x
    upper = max_x + 0.0000001 * std_x

    if symmetric_center is not None:
        assert lower < symmetric_center < upper, "symmetric_center is not in between [min, max]"

        left_interval = symmetric_center - lower
        right_interval = upper - symmetric_center

        interval = np.max([left_interval, right_interval])

        lower = symmetric_center - interval
        upper = symmetric_center + interval

    bin_edges = np.linspace(lower, upper, nbins + 1)

    return bin_edges

def equal_sample_bins(list_of_data, nbins):
    """
    Compute equal sample bins for the data

    Parameters:
    list_of_data: list
        list of data to compute bin edges
    nbins: int
        number of bins

    Returns:
    bin_edges: ndarray with shape (nbins+1,)
        bin edges
    """
    assert isinstance(list_of_data, list), "list_of_data must be a list"

    all_data = np.concatenate([data[:].ravel() for data in list_of_data])
    percents = np.linspace(0, 100., nbins + 1)
    bin_edges = np.percentile(all_data, percents)

    std_x = all_data.std()

    bin_edges[0] = bin_edges[0] - 0.00001 * std_x
    bin_edges[-1] = bin_edges[-1] + 0.00001 * std_x
    return bin_edges

def right_wrap(z, symm_center):
    """
    Reflect the data with respect to the symmetric center
    Parameters:
    z: ndarray with shape (N, T)
        data to reflect
    symm_center: float
        center of reflection

    Returns:
    new_z: ndarray with shape (N, T)
        reflected data
    """
    new_z = np.copy(z)
    where_to_apply = (new_z > symm_center)
    new_z[where_to_apply] = 2*symm_center - new_z[where_to_apply]
    return new_z

def left_wrap(z, symm_center): 
    """
    Reflect the data with respect to the symmetric center
    Parameters:
    z: ndarray with shape (N, T)
        data to reflect
    symm_center: float
        center of reflection

    Returns:
    new_z: ndarray with shape (N, T)
        reflected data
    """
    new_z = np.copy(z)
    where_to_apply = (new_z < symm_center)
    new_z[where_to_apply] = 2*symm_center - new_z[where_to_apply]
    return new_z

def right_replicate_fe(first_half):
    """
    Replicate the free energy to the right

    Parameters:
    first_half: ndarray with shape (nbins,)
        free energy of the first half

    Returns:
    second_half: ndarray with shape (nbins,)
        free energy of the second half, stacked to the right of first half
    """
    center = first_half[-1]
    second_half = 2*center - first_half[:-1]
    second_half = second_half[::-1]
    return np.hstack([first_half, second_half])

def left_replicate_fe(first_half):
    """
    Replicate the free energy to the left

    Parameters:
    first_half: ndarray with shape (nbins,)
        free energy of the first half

    Returns:
    second_half: ndarray with shape (nbins,)
        free energy of the second half, stacked to the left of the first half
    """
    center = first_half[0]
    second_half = 2*center - first_half[1:]
    second_half = second_half[::-1]
    return np.hstack([second_half, first_half])

def right_replicate_pmf(first_half):
    """
    Replicate the pmf to the right

    Parameters:
    first_half: ndarray with shape (nbins,)
        pmf of the first half

    Returns:
    second_half: ndarray with shape (nbins,)
        pmf of the second half, stacked to the right of first half
    """
    center = first_half[-1]
    second_half = 2*center - first_half
    second_half = second_half[::-1]
    return np.hstack([first_half, second_half])

def left_replicate_pmf(first_half):
    """
    Replicate the pmf to the left

    Parameters:
    first_half: ndarray with shape (nbins,)
        pmf of the first half

    Returns:
    second_half: ndarray with shape (nbins,)
        pmf of the second half, stacked to the left of the first half
    """
    center = first_half[0]
    second_half = 2*center - first_half
    second_half = second_half[::-1]
    return np.hstack([second_half, first_half])

def stride_lambda_indices(lambda_F, lambda_R, n):
    """
    Compute the indices for the stride lambda. Calculate indices_F and indices_R such that
    lambda_F[indices_F] == lambda_R[indices_R][::-1] or 
    lambda_F[indices_F][::-1] == lambda_R[indices_R]

    Parameters:
    lambda_F: ndarray with shape (NF,)
        lambda values of the forward direction
    lambda_R: ndarray with shape (NR,)
        lambda values of the reverse direction
    n: int
        number of indices

    Returns:
    indices_F: ndarray with shape (n,)
        indices of lambda_F
    indices_R: ndarray with shape (n,)
        indices of lambda_R
    """
    indices_F = np.linspace(0, lambda_F.shape[0] - 1, n)
    indices_F = np.round(indices_F)
    indices_F = indices_F.astype(np.int)

    indices_R = lambda_R.shape[0] - 1 - indices_F
    indices_R = indices_R[::-1]

    if not np.allclose(lambda_F[indices_F], lambda_R[indices_R][::-1]):
        raise IndexError("The condition lambda_F[indices_F] == lambda_R[indices_R][::-1] is not satisfied.")
    return indices_F, indices_R

def _close_where(scalar, array):
    replicated = np.array([scalar] * array.shape[0])
    near_zero = np.abs(array - replicated)
    return np.argmin(near_zero)


def closest_sub_array(source, reference, threshold=1e-3):
    """
    Find the indices of the closest elements in source to the reference elements
    Parameters:
    source: ndarray with shape (N,)
        source array
    reference: ndarray with shape (M,)
        reference array
    threshold: float
        threshold for the difference between the source and reference elements

    Returns:
    indices: ndarray with shape (M,)
        indices of the closest elements in source to the reference elements

    Raises:
    IndexError: if the difference between the source and reference elements is greater than the threshold
    """
    _source = copy.deepcopy(source)
    indices = []

    for i, ref_val in enumerate(reference):
        idx = _close_where(ref_val, _source)
        if np.abs(ref_val - source[idx]) > threshold:
            raise IndexError("element %d of array is too different from %d of ref"%(idx, i))
        indices.append(idx)
        _source[idx] = np.inf

    return np.array(indices)

def indices_F_to_R(indices_F, lambda_F, lambda_R):
    """
    Compute the indices_R from the indices_F such that
    lambda_F[indices_F] == lambda_R[indices_R][::-1]
    
    Parameters:
    indices_F: ndarray with shape (n,)
        indices of lambda_F
    lambda_F: ndarray with shape (NF,)
        lambda values of the forward direction
    lambda_R: ndarray with shape (NR,)
        lambda values of the reverse direction
    
    Returns:
    indices_R: ndarray with shape (n,)
        indices of lambda_R

    Raises:
    IndexError: if the condition lambda_F[indices_F] == lambda_R[indices_R][::-1] is not satisfied
    """
    indices_R = lambda_R.shape[0] - 1 - indices_F
    indices_R = indices_R[::-1]
    if not np.allclose(lambda_F[indices_F], lambda_R[indices_R][::-1]):
        raise IndexError("The condition lambda_F[indices_F] == lambda_R[indices_R][::-1] is not satisfied.")

    return indices_R