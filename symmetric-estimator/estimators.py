import numpy as np
from utils import hist_counts, bin_centers, center_reflection
from utils import bennett
from utils import time_reversal_of_work, time_reversal_of_trajectory

# Unidirectional estimator utilities

def uni_df_t(w_t):
    """
    unidirectional estimator for free energy difference

    Parameters: 
    w_t : 2d array
        work values for each trajectory

    Returns:
    df_t : 1d array
        free energy difference for each trajectory
    """
    assert w_t.ndim == 2, "w_t must be 2d array"

    exp_w_t = np.exp(-w_t)
    df_t = exp_w_t.mean(axis=0)
    df_t = -np.log(df_t)

    return df_t

def uni_pmf(z_t, w_t, lambda_t, V, ks, bin_edges):
    """
    unidirectional estimator for potential of mean force

    Parameters:
    z_t : 2d array
        trajectory data
    w_t : 2d array
        work values for each trajectory
    lambda_t : 1d array
        lambda values for each trajectory
    V : function
        potential energy function
    ks : 1d array  
        spring constants
    bin_edges : 1d array   
        bin edges for histogram

    Returns:
    centers : 1d array
        bin centers
    pmf : 1d array
        potential of mean force
    """
    assert z_t.ndim == w_t.ndim == 2, "z_t and w_t must be 2d array"
    assert z_t.shape == w_t.shape, "z_t and w_t must have the same shape"
    assert lambda_t.ndim == 1, "lambda_t  must be 1d array"
    assert z_t.shape[1] == lambda_t.shape[0], "z_t.shape[1], lambda_t.shape[0] must be the same"
    assert bin_edges.ndim == 1, "bin_edges must be 1d array"

    df_t = uni_df_t(w_t)

    N = z_t.shape[0]
    weights = np.exp(-w_t) / N
    histograms = hist_counts(z_t, bin_edges, weights)

    centers = bin_centers(bin_edges)
    denominator = np.exp( -V( centers[None,:], ks, lambda_t[:,None] ) + df_t[:,None] )
    denominator = denominator.sum(axis=0)

    numerator = histograms * np.exp(df_t[:,None])
    numerator = numerator.sum(axis=0)
    pmf = -np.log(numerator / denominator)

    return centers, pmf

# Bidirectional estimator utilities

def bi_df_t(wF_t, wR_t):
    """
    Bidirectional estimator for free energy difference

    Parameters:
    wF_t : 2d array
        forward work values for each trajectory
    wR_t : 2d array
        reverse work values for each trajectory

    Returns:
    df_t : 1d array
        free energy difference for each trajectory
    """
    assert wF_t.ndim == wR_t.ndim == 2, "wF_t and wR_t must be 2D array"
    assert wF_t.shape[1] == wR_t.shape[1], "number of steps in wF_t and wR_t must be the same"

    df_tau = bennett( wF_t[:, -1], wR_t[:, -1] )

    wR_t = time_reversal_of_work(wR_t)

    NF = wF_t.shape[0]
    NR = wR_t.shape[0]

    wF_tau = wF_t[:, -1]
    wR_tau = wR_t[:, -1]

    F_part = np.exp(-wF_t) / ( NF + NR * np.exp(df_tau -  wF_tau[:,None]) )
    F_part = F_part.sum(axis=0)

    R_part = np.exp(-wR_t) / ( NF + NR * np.exp(df_tau -  wR_tau[:,None]) )
    R_part = R_part.sum(axis=0)

    df_t = -np.log(F_part + R_part)

    return df_t

def bi_pmf(zF_t, wF_t, zR_t, wR_t, lambda_t, V, ks, bin_edges):
    """
    Bidirectional estimator for potential of mean force

    Parameters:
    zF_t : 2d array
        forward trajectory data
    wF_t : 2d array
        forward work values for each trajectory
    zR_t : 2d array
        reverse trajectory data
    wR_t : 2d array
        reverse work values for each trajectory
    lambda_t : 1d array
        lambda values for each trajectory
    V : function
        potential energy function
    ks : 1d array
        spring constants
    bin_edges : 1d array 
        bin edges for histogram

    Returns:
    centers : 1d array
        bin centers
    pmf : 1d array
        potential of mean force
    """
    assert zF_t.ndim == zR_t.ndim == wF_t.ndim == wR_t.ndim == 2, " zF_t, wF_t, zR_t, wR_t must be 2D array"
    assert zF_t.shape == wF_t.shape, "zF_t and wF_t must have the same shape"
    assert zR_t.shape == wR_t.shape, "zR_t and wR_t must have the same shape"
    assert lambda_t.ndim == 1, "lambda_t must be 1d array"
    assert zF_t.shape[1] == zR_t.shape[1] == lambda_t.shape[0], "zF_t.shape[1], zR_t.shape[1], lambda_t.shape[0] must be the same"
    assert bin_edges.ndim == 1, "bin_edges must be 1d array"

    df_t = bi_df_t(wF_t, wR_t)

    wR_t = time_reversal_of_work(wR_t)
    zR_t = time_reversal_of_trajectory(zR_t)

    NF = wF_t.shape[0]
    NR = wR_t.shape[0]

    wF_tau = wF_t[:, -1]
    wR_tau = wR_t[:, -1]
    dt_tau = df_t[-1]

    centers = bin_centers(bin_edges)

    denominator = np.exp(-V(centers[None, :], ks, lambda_t[:, None]) + df_t[:, None])
    denominator = denominator.sum(axis=0)

    weights_F = np.exp(-wF_t) / (NF + NR * np.exp(dt_tau - wF_tau[:, None]))
    histograms_F = hist_counts(zF_t, bin_edges, weights_F)

    weights_R = np.exp(-wR_t) / (NF + NR * np.exp(dt_tau - wR_tau[:, None]))
    histograms_R = hist_counts(zR_t, bin_edges, weights_R)

    numerator = (histograms_F + histograms_R) * np.exp(df_t[:, None])
    numerator = numerator.sum(axis=0)

    pmf = -np.log(numerator / denominator)
    return centers, pmf

# Version 1 of symmetric estimators for free energy and PMF.
# The symmetric estimator of path ensemble average is
#                         <F> = sum( F[x_n] + F[xr_n] * exp( -w_tau[x_n] ) ) / sum( 1 + exp( -w_tau[x_n] ) )
#                          where xr_n is the time reversal of x_n

def sym_est_df_t_v1(w_t):
    """
    Version 1 of symmetric estimator for free energy difference
    The symmetric estimator of path ensemble average is
    <F> = sum( F[x_n] + F[xr_n] * exp( -w_tau[x_n] ) ) / sum( 1 + exp( -w_tau[x_n] ) )
    where xr_n is the time reversal of x_n

    Parameters:
    w_t : 2d array
        work values for each trajectory

    Returns:
    df_t : 1d array
        free energy difference for each trajectory
    """
    assert w_t.ndim == 2, "w_t must be 2d array"

    w_tau = w_t[:, -1]
    wTR_t = time_reversal_of_work(w_t)

    denominator = (1 + np.exp(-w_tau)).sum()

    numerator = np.exp(-w_t) + np.exp(-wTR_t) * np.exp(-w_tau[:,None])
    numerator = numerator.sum(axis=0)

    df_t = -np.log(numerator / denominator)

    return df_t

def sym_est_pmf_v1(z_t, w_t, lambda_t, V, ks, bin_edges, symmetrize_pmf):
    """
    Version 1 of symmetric estimator for potential of mean force
    The symmetric estimator of path ensemble average is
    <F> = sum( F[x_n] + F[xr_n] * exp( -w_tau[x_n] ) ) / sum( 1 + exp( -w_tau[x_n] ) )
    where xr_n is the time reversal of x_n

    Parameters:
    z_t : 2d array
        trajectory data
    w_t : 2d array
        work values for each trajectory
    lambda_t : 1d array
        lambda values for each trajectory
    V : function
        potential energy function
    ks : 1d array 
        spring constants
    bin_edges : 1d array

    Returns:
    centers : 1d array
        bin centers
    pmf : 1d array
        potential of mean force
    """
    assert z_t.ndim == w_t.ndim == 2, "z_t and w_t must be 2d array"
    assert z_t.shape == w_t.shape, "z_t and w_t must have the same shape"
    assert lambda_t.ndim == 1, "lambda_t  must be 1d array"
    assert z_t.shape[1] == lambda_t.shape[0], "z_t.shape[1] and lambda_t.shape[0] must be the same"
    assert bin_edges.ndim == 1, "bin_edges must be 1d array"

    df_t  = sym_est_df_t_v1(w_t)

    wTR_t = time_reversal_of_work(w_t)
    zTR_t = time_reversal_of_trajectory(z_t)

    if symmetrize_pmf:
        symm_center = (lambda_t[0] + lambda_t[-1])/2.
        zTR_t = center_reflection(zTR_t, symm_center)

    centers = bin_centers(bin_edges)

    outer_denominator = np.exp( -V( centers[None,:], ks, lambda_t[:,None] ) + df_t[:,None] )
    outer_denominator = outer_denominator.sum(axis=0)

    w_tau = w_t[:, -1]
    inner_denominator = (1 + np.exp(-w_tau)).sum()

    weights_1 = np.exp(-w_t) / inner_denominator
    histograms_1 = hist_counts(z_t, bin_edges, weights_1)

    weights_2 = np.exp(-wTR_t) * np.exp(-w_tau[:,None]) / inner_denominator
    histograms_2 = hist_counts(zTR_t, bin_edges, weights_2)

    numerator = (histograms_1 + histograms_2) * np.exp(df_t[:,None])
    numerator = numerator.sum(axis=0)
    pmf = -np.log(numerator / outer_denominator)

    return centers, pmf
