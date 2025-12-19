import sys
from math import ceil
import numpy as np
import scipy.stats as sp
from scipy.spatial.distance import jensenshannon
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import diags, eye
from scipy.stats import spearmanr
from scipy.signal import convolve
import time
import logging
from typing import Dict, Tuple, Union
import os
import multiprocessing as mp

# from .util import * # Need for `emd` but is now deprecated

log = logging.getLogger()
log_bin = logging.getLogger('bin')

PEAK_MAX_VALUE_INDEX = 3
MAX_USHRT = 65535

def emd(
    p: np.ndarray,
    q: np.ndarray
) -> Tuple[float, float]:
    """
    DEPRECATED
    Finds the Earth Mover's Distance of two 1D arrays

    Parameters
    ----------
    p : ndarray
    q : ndarray

    Returns
    -------
    tuple[float, float]
        The Earth Mover's Distance; the weight of this calculation
    """
    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 and q_sum == 0:
        return 0, 0

    if p_sum == 0:
        return q.size, q_sum

    if q_sum == 0:
        return p.size, p_sum

    p = p / p_sum
    q = q / q_sum
    return c_emd(p, q, p.size), q_sum + p_sum


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base=2
) -> float:
    """
    LEGACY CODE
    Finds the jensen-shannon divergence

    Parameters
    ----------
    p : np.ndarray[np.float64]
        Graph of sample1
    q : np.ndarray[np.float64]
        Graph of sample2
    base : int, optional
        Determines base to be used in calculating scipy.entropy (Default is 2)

    Returns
    -------
    float
        Jensen-Shannon divergence
    """
    assert p.size == q.size
    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 and q_sum == 0:
        return 0

    if p_sum == 0:
        return 1

    if q_sum == 0:
        return 1

    p, q = p / p_sum, q / q_sum
    # log.info(np.info(p))
    # log.info(np.info(q))
    # log.info(p[p > 0.00000001])

    m = (p + q) / 2
    return sp.entropy(p, m, base=base) / 2 + sp.entropy(q, m, base=base) / 2

def output_graph(g, n, parent_dir, chrom, sample_name, do_output=False, sparse=False):
    """
    Outputs the graphs as images for visual inspection
    """
    if not do_output:
        return
    
    if g is None and n is None:
        return

    if chrom == "chr1":
        if not sparse:
            # Only save if chromosome 1
            np.save(f'{parent_dir}/{sample_name}_{chrom}_graph.npy', g)
            np.save(f'{parent_dir}/{sample_name}_{chrom}_node_weights.npy', n)
        else:
            adjacency_matrices = np.array([mat.toarray() for mat in g])
            np.save(f'{parent_dir}/{sample_name}_{chrom}_graph.npy', adjacency_matrices)
            np.save(f'{parent_dir}/{sample_name}_{chrom}_node_weights.npy', n)


def construct_windows(
    chrom_size: int,
    window_size: int,
    window_stride: int,
    bin_size: int
):
    """
    Computes the window indices and locations for this chromosome

    Parameters
    ----------
    chrom_size : int
        The size of the chromosome
    window_size : int
        Size of sliding window
    window_stride : int
        Factor of window_size to stride (e.g. 2 means stride by window_size/2)
    bin_size : int
        Binning resolution
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        window_indices : np.ndarray
            A Mx2 array where M is the number of windows
            The start and end indices of each window in bin units
        window_locations : np.ndarray
            A Mx2 array where M is the number of windows
            The start and end locations of each window in base pairs
    """
    step = window_size // window_stride

    if step == 0:
        log.error(f'Window stride factor {window_stride} too large for given window size {window_size}')
        raise ValueError(f'Window stride factor {window_stride} too large for given window size {window_size}')

    if chrom_size < window_size:
        numb_windows = 1
    else:
        numb_windows = np.floor((chrom_size - window_size) / step).astype(int) + 1

        if (chrom_size - window_size) % step != 0:
            numb_windows += 1

    window_starts = np.array([step * k for k in range(numb_windows) if step * k < chrom_size])
    window_ends = np.array([ws + window_size for ws in window_starts])
    window_ends = np.clip(window_ends, a_min=None, a_max=chrom_size)

    # Ensure that the last window is the same size as all the others by shifting it left 
    if chrom_size >= window_size:
        # Issue is only applicable if the chromosome is at least one full window long
        problematic = (window_ends - window_starts) < window_size
        if np.any(problematic):
            window_starts[problematic] = chrom_size - window_size
            window_ends[problematic] = chrom_size

            # Remove duplicates that may have been created
            window_starts, unique_indices = np.unique(window_starts, return_index=True) # Sorted (numpy v2.0.2)
            window_ends = window_ends[unique_indices]

    # Lets do a final check to ensure all windows are the correct size
    common_size = window_ends[0] - window_starts[0]
    if np.any(window_ends - window_starts != common_size):
        log.error('Error in constructing windows: Not all windows are the correct size')
        raise ValueError('Error in constructing windows: Not all windows are the correct size')

    window_locations = np.vstack((window_starts, window_ends)).T

    window_start_idx = np.floor(window_starts / bin_size).astype(int)
    # window_end_idx = np.floor(window_ends / bin_size).astype(int) + 1
    num_bins_per_window = np.ceil(common_size / bin_size).astype(int)
    window_end_idx = window_start_idx + num_bins_per_window

    window_indices = np.vstack((window_start_idx, window_end_idx)).T

    return window_indices, window_locations


def _fgw_distance_worker(args):
    C1_i, C2_i, p_i, q_i, alpha, M_single, skip_window, N, scale_cost, mass = args
    if skip_window:
        return np.nan
    
    use_partial = mass < 1.0
    
    if scale_cost == "unitless":
        # Compute W and GW for unitless scaling
        if use_partial:
            gw = ot.gromov.partial_gromov_wasserstein2(
                C1=C1_i,
                C2=C2_i,
                p=p_i,
                q=q_i,
                loss_fun="square_loss",
                symmetric=True,
                tol=1e-5,
                numItermax=300,
                m=mass,
                warn=False,
            )
            w = ot.partial.partial_wasserstein2(p_i, q_i, M_single, m=mass)
        else:
            gw = ot.gromov.gromov_wasserstein2(
                C1=C1_i,
                C2=C2_i,
                p=p_i,
                q=q_i,
                loss_fun="square_loss",
                symmetric=True,
                tol_abs=1e-5,
                tol_rel=1e-5,
                max_iter=300,
            )
            w = ot.emd2(p_i, q_i, M_single)

        scale_factor = np.sqrt(w / gw) if gw > 0 else 1.0
        C1_scaled = C1_i * scale_factor
        C2_scaled = C2_i * scale_factor
    else:
        # Interpret scale_cost as a numeric value
        C1_scaled = C1_i * scale_cost
        C2_scaled = C2_i * scale_cost

    # Compute FGW
    if use_partial:
        return ot.gromov.partial_fused_gromov_wasserstein2(
            M=M_single,
            C1=C1_scaled,
            C2=C2_scaled,
            p=p_i,
            q=q_i,
            alpha=alpha,
            loss_fun="square_loss",
            symmetric=True,
            tol=1e-5,
            numItermax=300, 
            m=mass, # only transport this amount of mass
            warn=False,
        )
    return ot.gromov.fused_gromov_wasserstein2(
        M=M_single,
        C1=C1_scaled,
        C2=C2_scaled,
        p=p_i,
        q=q_i,
        alpha=alpha,
        loss_fun="square_loss",
        symmetric=True,
        tol_abs=1e-5,
        tol_rel=1e-5,
        max_iter=300,
    )

def apply_mean_filter(A, ks=3):
    """ Helper function for GSP methods """
    return convolve(A, np.ones((ks, ks)) / (ks * ks), mode='same')

def pmf(x):
    """ Helper function for GSP methods """
    np.clip(x, a_min=0, a_max=None, out=x)
    x /= x.sum()
    return x

def random_walk(A, p, tol=1):
    """ Helper function for GSP methods """
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv = np.zeros_like(deg, dtype=A.dtype)
    nonzero = deg > tol
    deg_inv[nonzero] = 1.0 / deg[nonzero]
    P = A * deg_inv[None, :]

    # New: make the transition matrix disconnected if node is isolated
    isolated_nodes = ~nonzero
    P[:, isolated_nodes] = 0 # Only zero out the corresponding conditional PMF
    # Use np.diag_indices or explicit indexing for diagonal
    isolated_indices = np.where(isolated_nodes)[0]
    P[isolated_indices, isolated_indices] = 1.0
    return np.linalg.matrix_power(P, p)



def compare_signals(a, b, compare_method):
    """ Helper function for GSP methods """
    if compare_method == "spearman":
        score, _ = spearmanr(a, b)
        return score
    elif compare_method == "jsd":
        j_divergence = jensenshannon(a, b, base=2)
        # Map to [-1, 1] range and transform to similarity
        score = 2 * (1 - j_divergence) - 1 
        return score
    else:
        raise ValueError(f'Unknown compare method: {compare_method}')

def subsample(A1 : np.ndarray, A2 : np.ndarray, B1 : np.ndarray, B2: np.ndarray, p=1.0):
    '''
    Subsamples adjacency matrices A1, A2 and binding intensities B1, B2
    
    A1, A2: Input adjacency matrices with integer elements.
    B1, B2: Input binding intensities with integer elements. 
    p: After initial subsampling, further subsample to this level. 

    Returns: Updated A1, A2, B1, B2

    Author: Joseph Jackson
    '''
    def subsample_pair(A : np.ndarray, B : np.ndarray):
        depthA, depthB = A.sum(), B.sum()
            
        if depthA == 0 or depthB == 0:
            return A, B

        if depthA > depthB:
            A = np.random.binomial(A, depthB / depthA)
        else:
            B = np.random.binomial(B, depthA / depthB)
        return A, B
    
    A1_sub, A2_sub = subsample_pair(np.rint(A1).astype(np.int64), np.rint(A2).astype(np.int64))
    B1_sub, B2_sub = subsample_pair(np.rint(B1).astype(np.int64), np.rint(B2).astype(np.int64))
    if (p < 1.0):
        A1_sub = np.random.binomial(A1_sub, p)
        A2_sub = np.random.binomial(A2_sub, p)
        B1_sub = np.random.binomial(B1_sub, p)
        B2_sub = np.random.binomial(B2_sub, p)

    # Preserve data types
    A1_sub = A1_sub.astype(A1.dtype)
    A2_sub = A2_sub.astype(A2.dtype)
    B1_sub = B1_sub.astype(B1.dtype)
    B2_sub = B2_sub.astype(B2.dtype)

    return A1_sub, A2_sub, B1_sub, B2_sub


def _random_walk_gsp_worker(args):
    """ Computes random walk GSP """
    A1, x1, A2, x2, p, compare_method, cross, skip_window, ssp = args

    if ssp is not None:
        A1 = A1.toarray()
        A2 = A2.toarray()
        A1, A2, x1, x2 = subsample(A1, A2, x1, x2, p=ssp)
    
    sum_x1 = np.sum(x1)
    sum_x2 = np.sum(x2)
    max_A1 = np.max(A1)
    max_A2 = np.max(A2)

    both_zero = (np.isclose(sum_x1, 0) and np.isclose(sum_x2, 0)) or (np.isclose(max_A1, 0) and np.isclose(max_A2, 0))
    one_zero = (np.isclose(sum_x1, 0) or np.isclose(sum_x2, 0)) or (np.isclose(max_A1, 0) or np.isclose(max_A2, 0))
    if both_zero:
        return np.nan
    if one_zero:
        if compare_method == "spearman":
            return 0.0
        elif compare_method == "jsd":
            return 1.0 
        
    # Skip window MUST be here because there are some more special cases
    # that the `skip_window` check did not consider first
    if skip_window:
        return np.nan
    
    # Only make into dense if necessary
    if ssp is None:
        A1 = A1.toarray()
        A2 = A2.toarray()

    # Blur
    np.clip(apply_mean_filter(A1, ks=3), 0, None, out=A1)
    np.clip(apply_mean_filter(A2, ks=3), 0, None, out=A2)

    # Need to add self-loops for random walk
    # I = np.eye(A1.shape[0], dtype=A1.dtype)
    # A1 += I
    # A2 += I

    p = int(p)
    K1 = random_walk(A1, p)
    K2 = random_walk(A2, p)

    # Preprocess input signals to be PMFs
    x1 = pmf(x1)
    x2 = pmf(x2)

    # Forward pass i.e. K @ x
    x11 = K1 @ x1
    x22 = K2 @ x2

    # Numerical instability may not guarantee output is PMF
    x11 = pmf(x11)
    x22 = pmf(x22)

    if cross:
        # cross terms
        x12 = K2 @ x1
        x21 = K1 @ x2

        x12 = pmf(x12)
        x21 = pmf(x21)

        val_a = compare_signals(x11, x12, compare_method)
        val_b = compare_signals(x22, x21, compare_method)

        return 0.5 * (val_a + val_b)

    # Direct 
    return compare_signals(x11, x22, compare_method)

    
def laplacian_sparse(A, laplacian_type):
    """ 
    Helper function for GSP methods
    A is a CSR matrix (deprecated) 
    """
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    if laplacian_type == 'combinatorial':
        D = diags(deg, shape=(n, n), format='csr')
        L = D - A
        return 0.5 * (L + L.T)  # Ensure symmetry
    elif laplacian_type == "random_walk":
        # I - A D^-1
        deg_inv = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > 0
        deg_inv[nonzero] = 1.0 / deg[nonzero]
        D_inv = diags(deg_inv, shape=(n, n), format='csr')
        P = A @ D_inv # A D^{-1} make column stochastic
        I = eye(n, dtype=A.dtype, format='csr')
        return I - P
    
    elif laplacian_type == "symmetric_normalized":
        deg_inv_sqrt = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > 0
        deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        D_inv_sqrt = diags(deg_inv_sqrt, 0, shape=(n, n), format='csr')
        I = eye(n, dtype=A.dtype, format='csr')
        L = I - (D_inv_sqrt @ A @ D_inv_sqrt)
        return 0.5 * (L + L.T)  # ensure symmetry, stays sparse

    raise ValueError(f'Unknown laplacian type: {laplacian_type}')


def laplacian(A, laplacian_type, tol=1):
    """ 
    Helper function for GSP methods
    A is a dense matrix 
    """
    n = A.shape[0]
    deg = A.sum(axis=1) # will be a (n,) array
    if laplacian_type == 'combinatorial':
        D = np.diag(deg)
        L = D - A
        return 0.5 * (L + L.T)  # Ensure symmetry
    elif laplacian_type == "random_walk":
        # I - A D^-1
        deg_inv = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > tol
        deg_inv[nonzero] = 1.0 / deg[nonzero]
        P = A * deg_inv[None, :] # column stochastic (right mat mul)
        return np.eye(n, dtype=A.dtype) - P
    
    elif laplacian_type == "symmetric_normalized":
        deg_inv_sqrt = np.zeros_like(deg, dtype=A.dtype)
        nonzero = deg > tol
        deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        S = (deg_inv_sqrt[:, None] * A) * deg_inv_sqrt[None, :]  # D^{-1/2} A D^{-1/2}
        L = np.eye(n, dtype=A.dtype) - S

        # New: make the Laplacian disconnected where graph is disconnected
        isolated_nodes = ~nonzero
        L[isolated_nodes, :] = 0
        L[:, isolated_nodes] = 0

        return 0.5 * (L + L.T)  # Ensure symmetry
    raise ValueError(f'Unknown laplacian type: {laplacian_type}')

def construct_diffusion_kernel(L, t):
    """ Helper function for GSP methods """
    w, U = np.linalg.eigh(L)
    # Clip small negative eigenvalues to zero since L should be PSD
    np.clip(w, a_min=0, a_max=None, out=w) 
    exp_eigs = np.exp(-t * w)
    K = (U * exp_eigs) @ U.T
    return K


def _diffusion_gsp_worker(args):
    """ Computes diffusion GSP """
    A1, x1, A2, x2, t, compare_method, cross, skip_window, ssp = args

    if ssp is not None:
        A1 = A1.toarray()
        A2 = A2.toarray()
        A1, A2, x1, x2 = subsample(A1, A2, x1, x2, p=ssp)
    
    sum_x1 = np.sum(x1)
    sum_x2 = np.sum(x2)
    max_A1 = np.max(A1)
    max_A2 = np.max(A2)

    both_zero = (np.isclose(sum_x1, 0) and np.isclose(sum_x2, 0)) or (np.isclose(max_A1, 0) and np.isclose(max_A2, 0))
    one_zero = (np.isclose(sum_x1, 0) or np.isclose(sum_x2, 0)) or (np.isclose(max_A1, 0) or np.isclose(max_A2, 0))
    if both_zero:
        return np.nan
    if one_zero:
        if compare_method == "spearman":
            return 0.0
        elif compare_method == "jsd":
            return 1.0 
        
    # Skip window MUST be here because there are some more special cases
    # that the `skip_window` check did not consider first
    if skip_window:
        return np.nan
        
    # If not already dense, convert to dense for blurring
    if ssp is None:
        A1 = A1.toarray()
        A2 = A2.toarray()

    # Blur
    np.clip(apply_mean_filter(A1, ks=3), 0, None, out=A1)
    np.clip(apply_mean_filter(A2, ks=3), 0, None, out=A2)

    # Compute kernels
    L1 = laplacian(A1, laplacian_type="symmetric_normalized")
    L2 = laplacian(A2, laplacian_type="symmetric_normalized")

    # K1 = construct_diffusion_kernel(L1, t) # Takes a long time
    # K2 = construct_diffusion_kernel(L2, t) # Takes a long time

    # Preprocess input signals to be PMFs
    x1 = pmf(x1)
    x2 = pmf(x2)

    # Forward pass i.e. exp^{-tL} @ x
    x11 = expm_multiply(-t * L1, x1)  
    x22 = expm_multiply(-t * L2, x2)  
    # x11 = K1 @ x1 # Takes a long time
    # x22 = K2 @ x2 # Takes a long time

    # Numerical instability may not guarantee output is PMF
    x11 = pmf(x11)
    x22 = pmf(x22)

    if cross:
        # cross terms
        x12 = expm_multiply(-t * L2, x1)
        x21 = expm_multiply(-t * L1, x2) 
        # x12 = K2 @ x1 # Takes a long time
        # x21 = K1 @ x2 # Takes a long time

        x12 = pmf(x12)
        x21 = pmf(x21)

        val_a = compare_signals(x11, x12, compare_method)
        val_b = compare_signals(x22, x21, compare_method)

        return 0.5 * (val_a + val_b)

    # Direct 
    return compare_signals(x11, x22, compare_method)
    


def _wasserstein_worker(args):
    x1, x2, skip_window, feat, weight = args
    if skip_window:
        return np.nan
    
    n = x1.shape[0]

    if weight == "uniform":
        p = np.full(n, 1.0 / n)
        q = np.full(n, 1.0 / n)
    elif weight == "BA":
        sum_x1 = np.sum(x1)
        sum_x2 = np.sum(x2)
        p = x1 / sum_x1 if sum_x1 > 0 else np.full(n, 1.0 / n)
        q = x2 / sum_x2 if sum_x2 > 0 else np.full(n, 1.0 / n)
    else:
        p = np.full(n, 1.0 / n)
        q = np.full(n, 1.0 / n)


    def compute_normalize_features(features):
        mean_ = np.mean(features, axis=0, keepdims=True)
        mean_[mean_ == 0] = 1.0
        return features / mean_

    # Feature construction
    if feat == "index":
        # assume weight is BA 
        return c_emd(p.astype(np.float64), q.astype(np.float64), n)
    elif feat == "BA":
        X1 = x1.reshape(-1, 1).astype(np.float32)
        X2 = x2.reshape(-1, 1).astype(np.float32)
    elif feat == "index_BA":
        X1 = np.vstack((np.arange(n), x1)).T.astype(np.float32)
        X2 = np.vstack((np.arange(n), x2)).T.astype(np.float32)

    # Mean normalize
    X1 = compute_normalize_features(X1)
    X2 = compute_normalize_features(X2)

    M = ot.dist(X1, X2, metric="euclidean")

    return ot.emd2(p, q, M)


def _jsd_worker(args):
    g1, g2 = args

    max_graph = np.max(g1)
    max_o_graph = np.max(g2)

    if max_graph == 0 and max_o_graph == 0:
        return 0.0

    if max_graph == 0 or max_o_graph == 0:
        return -1.0

    # Blur
    np.clip(apply_mean_filter(g1, ks=3), 0, None, out=g1)
    np.clip(apply_mean_filter(g2, ks=3), 0, None, out=g2)

    graph_flat = g1.flatten()
    o_graph_flat = g2.flatten()
    j_divergence = jensen_shannon_divergence(graph_flat, o_graph_flat)

    # Make j_value range from -1 to 1
    j_value = 2 * (1 - j_divergence) - 1

    return j_value

def _emd_worker(args):
    g1, g2 = args

    max_graph = np.max(g1)
    max_o_graph = np.max(g2)

    if max_graph == 0 and max_o_graph == 0:
        return 0.0

    if max_graph == 0 or max_o_graph == 0:
        return -1.0

    # Blur
    np.clip(apply_mean_filter(g1, ks=3), 0, None, out=g1)
    np.clip(apply_mean_filter(g2, ks=3), 0, None, out=g2)
    
    # Calculate emd for all rows and columns -> Take weighted average
    emd_distance_list = []
    emd_weight_list = []
    for k in range(g1.shape[0]):
        emd_dist, emd_weight = emd(g1[k], g2[k])
        emd_distance_list.append(emd_dist)
        emd_weight_list.append(emd_weight)

        emd_dist, emd_weight = emd(g1[:, k], g2[:, k])
        emd_distance_list.append(emd_dist)
        emd_weight_list.append(emd_weight)

    max_emd_weight = np.max(emd_weight_list)

    if max_emd_weight == 0:
        overall_emd_dist = 0.0
    else:
        overall_emd_dist = np.average(emd_distance_list, weights=emd_weight_list)

    # Transformation
    max_emd_dist = g1.shape[0] - 1
    numerator = overall_emd_dist - max_emd_dist
    emd_value = 2 * numerator * numerator / (
            max_emd_dist * max_emd_dist) - 1
    
    return emd_value


    

class ChromBinData:
    """
    A class used to represent a chromosome in a sample

    The difference with `ChromLoopData` class is that this class is a binned representation
    of loops, as opposed to loop-level granularity

    Attributes
    ----------
    name : str
        The name of the chromosome
    size : int
        The size of the chromosome
    sample_name : str
        The name of the sample this chromosome belongs to
    bin_size : int
        The size of each bin i.e. resolution
    
    window_indices : np.ndarray
        The start and end indices of each window in bin units
    window_locations : np.ndarray
        The start and end locations of each window in base pairs
    adjacency_matrices : list
        A list of NxN COO sparse adjacency matrices for each window.
        N is the number of bins per window.
    node_weights : np.ndarray
        A MxN array of binding affinity for the chromosome
        M is the number of windows
        N is the number of bins per window
    """

    def __init__(
            self,
            chrom_name: str,
            chrom_size: int,
            sample_name: str,
            bin_size: int,
            window_size: int,
            window_stride: int
    ):
        # Parameters
        self.name = chrom_name
        self.size = chrom_size
        self.sample_name = sample_name
        self.bin_size = bin_size

        # Data Attributes
        self.window_indices, self.window_locations = construct_windows(chrom_size, window_size, window_stride, bin_size)
        self.adjacency_matrices = None # Populated later on
        self.node_weights = None # Populated later on

    def finish_init(self):
        """
        Finalize the initialization of this chromosome data
        Checks the sizes of each attributes. Specifically that 

            - self.adjacency_matrices is a list of M sparse COO matrices that are (N, N) 
            - self.node_weights is shape (M, N)
        
        where M is the number of windows and N is the number of bins per window
        """
        num_windows = self.window_indices.shape[0]
        num_bins_per_window = self.window_indices[0, 1] - self.window_indices[0, 0]

        if self.adjacency_matrices is not None:
            if len(self.adjacency_matrices) != num_windows:
                log.error(f'Adjacency matrices length mismatch for {self.sample_name} {self.name}: '
                          f'expected {num_windows}, got {len(self.adjacency_matrices)}')
                raise ValueError(f'Adjacency matrices length mismatch for {self.sample_name} {self.name}: '
                                 f'expected {num_windows}, got {len(self.adjacency_matrices)}')

            for i, mat in enumerate(self.adjacency_matrices):
                if mat.shape != (num_bins_per_window, num_bins_per_window):
                    log.error(f'Adjacency matrix shape mismatch for {self.sample_name} {self.name} window {i}: '
                              f'expected ({num_bins_per_window}, {num_bins_per_window}), '
                              f'got {mat.shape}')
                    raise ValueError(f'Adjacency matrix shape mismatch for {self.sample_name} {self.name} window {i}: '
                                     f'expected ({num_bins_per_window}, {num_bins_per_window}), '
                                     f'got {mat.shape}')

        if self.node_weights is not None:
            if self.node_weights.shape != (num_windows, num_bins_per_window):
                log.error(f'Node weights shape mismatch for {self.sample_name} {self.name}: '
                          f'expected ({num_windows}, {num_bins_per_window}), '
                          f'got {self.node_weights.shape}')
                raise ValueError(f'Node weights shape mismatch for {self.sample_name} {self.name}: '
                                 f'expected ({num_windows}, {num_bins_per_window}), '
                                 f'got {self.node_weights.shape}')
            

    def filter_with_peaks(
        self,
        peak_array: np.ndarray,
        both_peak_support: bool = False
    ) -> bool:
        """
        Filters out Hi-C contacts without peak support using matrix operations

        Parameters
        ----------
        peak_array : np.ndarray
            1D array of peak values (dense)
        both_peak_support : bool, optional
            Whether to only keep loops that have peak support on both sides
            (default is False)

        Returns
        -------
        bool
            Whether the chromosome had any problems when filtering
        """
        start_time = time.time()
        
        # Count non-zero peaks for logging
        nnz = np.count_nonzero(peak_array)
        log.info(f"Filtering {self.sample_name} {self.name} with {nnz}/{len(peak_array)} number of non-zero bins (approx no. peaks)...")

        total_before = 0
        total_after = 0
        # Iterate over windows and apply filter
        for i, adj_mat in enumerate(self.adjacency_matrices):
            # Get window bounds
            window_start_idx = self.window_indices[i, 0]
            window_end_idx = self.window_indices[i, 1]
            
            # Extract peaks for this window
            # Ensure we don't go out of bounds of peak_array
            p_window = peak_array[window_start_idx:window_end_idx]
            
            # Create binary mask (1 if peak, 0 if not)
            mask = (p_window > 0).astype(np.float32)
            # Float32 aligns with the adjacency matrix datatype
            
            # Create diagonal matrix
            D = diags(mask)
            
            if both_peak_support:
                # Keep interaction if BOTH anchors are peaks
                # A_new = D * A * D
                self.adjacency_matrices[i] = D @ adj_mat @ D
            else:
                # Keep interaction if AT LEAST ONE anchor is a peak
                # Equivalent to removing interactions where NEITHER is a peak
                # A_removed = (I-D) * A * (I-D)
                # A_new = A - A_removed
                
                inv_mask = 1.0 - mask
                D_inv = diags(inv_mask)
                
                # Calculate what to remove (interactions between two non-peaks)
                to_remove = D_inv @ adj_mat @ D_inv
                
                # Subtract from original
                self.adjacency_matrices[i] = adj_mat - to_remove
            
            # Clean up zeros to maintain sparsity
            self.adjacency_matrices[i].eliminate_zeros()

            total_before += adj_mat.data.sum()
            total_after += self.adjacency_matrices[i].data.sum()

        # Check if we have any loops left
        log.info(f"Genome-wide sum of adjacency matrix weights:"
                 f"  Before peak support filter: {total_before}"
                 f"  After peak support filter: {total_after}"
                 )

        if total_after == 0:
            log.warning(f"No loops left. Skipping")
            return False

        log.debug(f'Time taken: {time.time() - start_time}\n')
        return True




    def fgw_distance_batch(self, g1, g2, x1, x2, alpha, cost, gvmax=100.0,
                           scale_cost='unitless', mass=1.0, num_cores=1):
        """
        DEPRECATED

        Computes the FGW on batches of attributed graphs
        Specifically, we batch across the first dimension (M):

            - g1: MxNxN
            - x1: MxN

        Returns a list of FGW distances for each pair in the batch and the window weights
        """
        # Ensure non-negative values first
        g1 = np.maximum(g1, 0)
        g2 = np.maximum(g2, 0)
        
        # Compute per-window max from the current arrays
        max_graph = np.max(g1, axis=(1, 2))
        o_max_graph = np.max(g2, axis=(1, 2))

        # Compute window weights
        global_max_graph = np.max(max_graph)
        global_max_o_graph = np.max(o_max_graph)
        
        denom_g1 = global_max_graph if global_max_graph > 0 else 1.0
        denom_g2 = global_max_o_graph if global_max_o_graph > 0 else 1.0
        window_weights = max_graph / denom_g1 + o_max_graph / denom_g2

        common_max = np.maximum(max_graph, o_max_graph)

        # Condition 1: both graphs are zero matrices
        graphs_zero = (common_max == 0)

        # Condition 2: more than 80% of node feature bins are zero
        # node_sparse_1 = (np.sum(x1 == 0, axis=1) / x1.shape[1]) > 0.8
        # node_sparse_2 = (np.sum(x2 == 0, axis=1) / x2.shape[1]) > 0.8
        # node_sparse = node_sparse_1 | node_sparse_2

        # Condition 3: both graphs after percentile clipping are zero matrices
        g1_flat = g1.reshape(g1.shape[0], -1)
        g2_flat = g2.reshape(g2.shape[0], -1)
        g1_vmax = np.percentile(g1_flat, gvmax, axis=1)
        g2_vmax = np.percentile(g2_flat, gvmax, axis=1)
        graphs_percentile_zero = (g1_vmax == 0) & (g2_vmax == 0)

        # skip_window = graphs_zero | node_sparse | graphs_percentile_zero
        skip_window = graphs_zero | graphs_percentile_zero # Remove condition 2

        # Create cost matrices using the already-computed max values
        if cost == "linear_pairs":
            denom = np.where(common_max > 0, common_max, 1.0)[:, None, None]
            C1 = 1.0 - (g1 / denom)
            C2 = 1.0 - (g2 / denom)
        elif cost == "linear":
            denom1 = np.where(max_graph > 0, max_graph, 1.0)[:, None, None]
            denom2 = np.where(o_max_graph > 0, o_max_graph, 1.0)[:, None, None]
            C1 = 1.0 - (g1 / denom1)
            C2 = 1.0 - (g2 / denom2)
        elif cost == "adjacency":
            denom1 = np.where(max_graph > 0, max_graph, 1.0)[:, None, None]
            denom2 = np.where(o_max_graph > 0, o_max_graph, 1.0)[:, None, None]
            C1 = g1 / denom1
            C2 = g2 / denom2
        else:
            log.error(f'Unknown cost function: {cost}')
            raise ValueError(f'Unknown cost function: {cost}')

        # Use node features as weights (PMF)
        x1_sum = x1.sum(axis=1, keepdims=True)
        x2_sum = x2.sum(axis=1, keepdims=True)

        # Handle cases where sum is 0 to avoid division by zero and create uniform distribution
        p = np.divide(x1, x1_sum, where=x1_sum > 0, out=np.full_like(x1, 1.0 / x1.shape[1]))
        q = np.divide(x2, x2_sum, where=x2_sum > 0, out=np.full_like(x2, 1.0 / x2.shape[1]))

        # Create linear distance decay feature matrix
        num_bins = g1.shape[1]
        coords = np.arange(num_bins).reshape(-1, 1)

        # The feature cost matrix M is the same for all items in the batch
        M_single = ot.dist(coords, coords, metric='euclidean')
        M_single = M_single.astype(np.float32) / np.max(M_single) 

        # Turns out to be slower on CPU for batch processing
        # res = ot.batch.solve_gromov_batch(
        #     C1=C1,
        #     C2=C2,
        #     a=p,
        #     b=q,
        #     M=M_single[None, :, :],
        #     alpha=alpha,
        #     loss="sqeuclidean",
        #     symmetric=True,
        # )
        # distances = res.value

        M = g1.shape[0]
        N = g1.shape[1]
        if num_cores > 1:
            # Multi-core processing
            tasks = [
                (C1[i], C2[i], p[i], q[i], alpha, M_single, skip_window[i], N, scale_cost, mass)
                for i in range(M)
            ]
            with mp.Pool(processes=num_cores) as pool:
                distances = np.array(pool.map(_fgw_distance_worker, tasks))
        else:
            # Fallback to single core 
            use_partial = mass < 1.0

            distances = []
            for i in range(M):
                if skip_window[i]:
                    distances.append(np.nan)
                else:
                    # Scale cost matrices
                    if scale_cost == "unitless":
                        # Compute W and GW for unitless scaling
                        if use_partial:
                            gw = ot.gromov.partial_gromov_wasserstein2(
                                C1=C1[i],
                                C2=C2[i],
                                p=p[i],
                                q=q[i],
                                loss_fun="square_loss",
                                symmetric=True,
                                tol=1e-5,
                                numItermax=300,
                                m=mass,
                                warn=False,
                            )
                            w = ot.partial.partial_wasserstein2(p[i], q[i], M_single, m=mass)
                        else:
                            gw = ot.gromov.gromov_wasserstein2(
                                C1=C1[i],
                                C2=C2[i],
                                p=p[i],
                                q=q[i],
                                loss_fun="square_loss",
                                symmetric=True,
                                tol_abs=1e-5,
                                tol_rel=1e-5,
                                max_iter=300,
                            )
                            w = ot.emd2(p[i], q[i], M_single)
                        
                        scale_factor = np.sqrt(w / gw) if gw > 0 else 1.0
                        C1_scaled = C1[i] * scale_factor
                        C2_scaled = C2[i] * scale_factor
                    else:
                        # Interpret scale_cost as a numeric value
                        C1_scaled = C1[i] * scale_cost
                        C2_scaled = C2[i] * scale_cost
                    
                    # Compute FGW
                    if use_partial:
                        dist = ot.gromov.partial_fused_gromov_wasserstein2(
                            M=M_single,
                            C1=C1_scaled,
                            C2=C2_scaled,
                            p=p[i],
                            q=q[i],
                            alpha=alpha,
                            loss_fun="square_loss",
                            symmetric=True,
                            tol=1e-5,
                            numItermax=300,
                            m=mass,
                            warn=False,
                        )
                    else:
                        dist = ot.gromov.fused_gromov_wasserstein2(
                            M=M_single,
                            C1=C1_scaled,
                            C2=C2_scaled,
                            p=p[i],
                            q=q[i],
                            alpha=alpha,
                            loss_fun="square_loss",
                            symmetric=True,
                            tol_abs=1e-5,
                            tol_rel=1e-5,
                            max_iter=300,
                        )
                    distances.append(dist)
            distances = np.array(distances)

        return distances, window_weights
    

    def gsp_batch(self, o_chrom, kernel_type, mu, compare_method, cross, ssp, num_cores=1):
        """
        Computes the GSP distances on batches of graphs.
        Uses sparse matrices directly for efficiency with expm_multiply.
        
        Parameters
        ----------
        o_chrom : ChromBinData
            The other chromosome to compare against
        kernel_type : str
            Type of graph kernel to use: either "diffusion" or "random_walk"
        mu : float or int
            Diffusion time parameter
            OR 
            Power for random walk
        compare_method : str
            Method to compare diffusion states (e.g., "spearman", "jsd")
        cross : bool
            Whether to use cross comparison
        ssp : float
            Subsample percentage in [0, 1]. If None, then no subsampling is done.
            If specified as a float between 0 and 1, then that initial subsampling is done
            to ensure the read depth is identical between any two pairs. 
            Additional subsampling is done that is a proportion of this common read depth.
        num_cores : int
            Number of cores to use
        """
        # We use the sparse matrices directly
        adj_list1 = self.adjacency_matrices
        adj_list2 = o_chrom.adjacency_matrices
        
        # Node weights are dense (M, N)
        weights1 = self.node_weights
        weights2 = o_chrom.node_weights
        
        M = len(adj_list1) # number of windows
        N = weights1.shape[1] # number of bins per window

        # BA is not yet normalized, so we deem BA to be "0" if count is less than 1
        # If more than 50% of the nodes in the window are "0", we skip the window
        node_sparse_1 = (np.sum(weights1 < 1, axis=1) / N) > 0.5
        node_sparse_2 = (np.sum(weights2 < 1, axis=1) / N) > 0.5
        skip_window = node_sparse_1 | node_sparse_2

        tasks = [
            (adj_list1[i], weights1[i].copy(), adj_list2[i], weights2[i].copy(), mu, compare_method, cross, skip_window[i], ssp)
            for i in range(M)
        ]

        if num_cores > 1:
            with mp.Pool(processes=num_cores) as pool:
                if kernel_type == "diffusion":
                    distances = np.array(pool.map(_diffusion_gsp_worker, tasks))
                elif kernel_type == "random_walk":
                    distances = np.array(pool.map(_random_walk_gsp_worker, tasks))
                else:
                    raise ValueError(f'Unknown kernel type: {kernel_type}')
        else:
            if kernel_type == "diffusion":
                distances = np.array([_diffusion_gsp_worker(t) for t in tasks])
            elif kernel_type == "random_walk":
                distances = np.array([_random_walk_gsp_worker(t) for t in tasks])
            else:
                raise ValueError(f'Unknown kernel type: {kernel_type}')

        window_weights = np.ones(M)

        return distances, window_weights
    
    def wasserstein_batch(self, o_chrom, feat, weight, num_cores=1):
        """
        DEPRECATED 
        
        Computes the Wasserstein distance on batches of graphs.  

        Parameters
        ----------
        o_chrom : ChromBinData
            The other chromosome to compare against
        num_cores : int
            Number of cores to use
        """
        weights1 = self.node_weights
        weights2 = o_chrom.node_weights

        # Number of windows
        M = weights1.shape[0]

        tasks = [
            (weights1[i], weights2[i], False, feat, weight)
            for i in range(M)
        ]

        if num_cores > 1:
            with mp.Pool(processes=num_cores) as pool:
                distances = np.array(pool.map(_wasserstein_worker, tasks))
        else:
            distances = np.array([_wasserstein_worker(t) for t in tasks])

        window_weights = np.ones(M)

        return distances, window_weights
    
    def emd_jsd_batch(self, o_chrom, method, num_cores=1):
        """
        Computes JSD or EMD on batches of graphs using the old logic.
        """
        # Convert to dense matrices
        adj_list1 = [mat.toarray() for mat in self.adjacency_matrices]
        adj_list2 = [mat.toarray() for mat in o_chrom.adjacency_matrices]

        # Number of windows
        M = len(adj_list1)

        # Compute window weights
        max_graph = np.array([np.max(m) for m in adj_list1])
        o_max_graph = np.array([np.max(m) for m in adj_list2])
        
        global_max_graph = np.max(max_graph)
        global_max_o_graph = np.max(o_max_graph)
        
        denom_g1 = global_max_graph if global_max_graph > 0 else 1.0
        denom_g2 = global_max_o_graph if global_max_o_graph > 0 else 1.0
        window_weights = max_graph / denom_g1 + o_max_graph / denom_g2

        tasks = [
            (adj_list1[i].astype(np.float64), adj_list2[i].astype(np.float64))
            for i in range(M)
        ]

        if num_cores > 1:
            if method == "emd":
                with mp.Pool(processes=num_cores) as pool:
                    distances = np.array(pool.map(_emd_worker, tasks))
            else: 
                with mp.Pool(processes=num_cores) as pool:
                    distances = np.array(pool.map(_jsd_worker, tasks))
        else:
            if method == "emd":
                distances = np.array([_emd_worker(t) for t in tasks])
            else:
                distances = np.array([_jsd_worker(t) for t in tasks])

        return distances, window_weights


    def compare(
        self,
        o_chrom: 'ChromBinData',
        alpha: float = 0.5,
        method: str = 'random_walk',
        mu: Union[float, int] = 1.0,
        cross: bool = False,
        compare_method: str = 'spearman',
        cost : str = 'linear',
        output_dir: str = 'output',
        do_output_graph: bool = False,
        param_str: str = '',
        gvmax: float = 100.0,
        scale_cost: str = 'unitless',
        mass: float = 1.0,
        feat: str = 'index_BA',
        weight: str = 'uniform',
        ssp: float = None,
        num_cores: int = 1
    ):
        """
        Compares this chromosome with another chromosome using the specified method

        The sparse matrices are all converted to dense matrices for batch comparison.
        Specifically:

        1. Adjacency matrices: MxNxN
        2. Node weights: MxN

        where M is the number of windows and N is the number of bins per window

        Parameters
        ----------
        o_chrom : ChromBinData
            The other chromosome to compare against
        method : str
            Comparison method to use. Options are:

            - 'random_walk'
            - 'diffusion'
        mu : float or int
            Power to raise the random walk transition matrix (if method is 'random_walk')
            Diffusion time step (if method is 'diffusion')    
        cross : bool
            Whether to compare the signals 'direct' or 'cross'.
            See chia_rep.compare() documentation for details.
        compare_method : str
            Method to compare processed binding affinity signals 
            after passing through random walk or diffusion.
            Either 'spearman' or 'jsd'
        output_dir : str
            Output folder
        do_output_graph : bool
            Whether to output the processed graphs for each window of chromosome 1
            as .npys for debugging purposes
        num_cores : int
            Number of pools to use for parallel processing across the windows of a chromosome
        param_str : str
            Parameter string that is the subfolder name under output_dir
        ssp : float
            Subsample percentage in [0, 1]. If None, then no subsampling is done.
            If specified as a float between 0 and 1, then that initial subsampling is done
            to ensure the read depth is identical between any two pairs. 
            Additional subsampling is done that is a proportion of this common read depth.

        Returns
        -------
        tuple
            A tuple containing:
            - distances : np.ndarray
                Array of distances for each window
            - window_weights : np.ndarray (deprecated)
                Array of weights for each window
            - max_graph : np.ndarray (deprecated)
                Array of maximum graph values for each window in this chromosome
            - o_max_graph : np.ndarray (deprecated)
                Array of maximum graph values for each window in the other chromosome
        """

        # Output graph variables if chr1
        comparison_name = f'{self.sample_name}_{o_chrom.sample_name}'
        parent_dir = f'{output_dir}/{param_str}/comparisons/{comparison_name}'

        if method == "fgw":

            # Convert sparse matrices to dense
            adjacency_matrices = np.array([mat.toarray() for mat in self.adjacency_matrices])
            node_weights = self.node_weights # already dense

            o_adjacency_matrices = np.array([mat.toarray() for mat in o_chrom.adjacency_matrices])
            o_node_weights = o_chrom.node_weights # already dense

            distances, window_weights = self.fgw_distance_batch(g1=adjacency_matrices,
                                            g2=o_adjacency_matrices,
                                            x1=node_weights,
                                            x2=o_node_weights,
                                            alpha=alpha,
                                            cost=cost,
                                            gvmax=gvmax,
                                            scale_cost=scale_cost,
                                            mass=mass,
                                            num_cores=num_cores,
                                            )
            
            output_graph(adjacency_matrices, node_weights, 
                         parent_dir, self.name, self.sample_name, do_output_graph, sparse=False)
            output_graph(o_adjacency_matrices, o_node_weights, 
                         parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph, sparse=False)

            # Recompute max_graph for return value 
            # Used for diagnostics
            max_graph = np.max(np.maximum(adjacency_matrices, 0), axis=(1, 2))
            o_max_graph = np.max(np.maximum(o_adjacency_matrices, 0), axis=(1, 2))
            result = (distances, window_weights, max_graph, o_max_graph)

        elif method in ["diffusion", "random_walk"]:

            distances, window_weights = self.gsp_batch(o_chrom, kernel_type=method, mu=mu, 
                                                       compare_method=compare_method, cross=cross, ssp=ssp,
                                                       num_cores=num_cores)

            output_graph(self.adjacency_matrices, self.node_weights, 
                         parent_dir, self.name, self.sample_name, do_output_graph, sparse=True)
            output_graph(o_chrom.adjacency_matrices, o_chrom.node_weights, 
                         parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph, sparse=True)

            # For DW, we don't have max_graphs in the same sense as FGW
            # Lets return array of same shape of nans
            nans = np.full_like(distances, np.nan)
            result = (distances, window_weights, nans, nans)

        elif method == "w":

            distances, window_weights = self.wasserstein_batch(o_chrom, feat=feat, weight=weight, num_cores=num_cores)

            output_graph(self.adjacency_matrices, self.node_weights, 
                         parent_dir, self.name, self.sample_name, do_output_graph, sparse=True)
            output_graph(o_chrom.adjacency_matrices, o_chrom.node_weights, 
                         parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph, sparse=True)

            nans = np.full_like(distances, np.nan)
            result = (distances, window_weights, nans, nans)

        elif method == "JSD":
            # Apply the old JSD distance function
            distances, window_weights = self.emd_jsd_batch(o_chrom, method="jsd", num_cores=num_cores)

            output_graph(self.adjacency_matrices, self.node_weights, 
                         parent_dir, self.name, self.sample_name, do_output_graph, sparse=True)
            output_graph(o_chrom.adjacency_matrices, o_chrom.node_weights, 
                         parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph, sparse=True)

            nans = np.full_like(distances, np.nan)
            result = (distances, window_weights, nans, nans)

        elif method == "EMD":
            # Apply the old EMD distance function
            distances, window_weights = self.emd_jsd_batch(o_chrom, method="emd", num_cores=num_cores)

            output_graph(self.adjacency_matrices, self.node_weights, 
                         parent_dir, self.name, self.sample_name, do_output_graph, sparse=True)
            output_graph(o_chrom.adjacency_matrices, o_chrom.node_weights, 
                         parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph, sparse=True)

            nans = np.full_like(distances, np.nan)
            result = (distances, window_weights, nans, nans)

        

        return result









    
        
        

