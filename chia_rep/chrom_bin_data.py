import sys
from math import ceil
import numpy as np
import scipy.stats as sp
from scipy.linalg import expm
from scipy.spatial.distance import cosine, jensenshannon
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import diags
import ot
import time
import logging
from pyBedGraph import BedGraph
from typing import Dict, Tuple, Union
import os
import multiprocessing as mp

from .util import *

log = logging.getLogger()
log_bin = logging.getLogger('bin')


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
    Constructs the window indices for this chromosome

    Parameters
    ----------
    chrom_size : int
        The size of the chromosome
    window_size : int
        The size of each window
    window_stride : int
        The stride between each window
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
                tol=1e-6,
                numItermax=1000,
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
                tol_abs=1e-6,
                max_iter=1000,
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
            tol=1e-6,
            numItermax=1000, # 1e3 (default is 1e4)
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
        tol_abs=1e-6,
        max_iter=1000,
    )

def _diffusion_wasserstein_worker(args):
    adj1, adj2, x1, x2, tau, skip_window = args
    if skip_window:
        return np.nan
    
    sum_x1 = np.sum(x1)
    sum_x2 = np.sum(x2)

    if sum_x1 == 0 and sum_x2 == 0:
        return 0.0
    
    n = x1.shape[0]

    # Helper to compute diffused signal
    def diffuse_signal(adj, x, t):
        # Compute combinatorial Laplacian L = D - A
        # adj is sparse
        degrees = np.array(adj.sum(axis=1)).flatten()
        D = diags(degrees)
        L = D - adj
        
        # Diffuse: x_tilde = exp(-tau * L) * x
        # We use expm_multiply for efficiency: exp(A)v
        # Here A = -t * L
        return expm_multiply(-t * L, x)
    
    def compute_normalized_features(features):
        sum_ = np.sum(features, axis=0, keepdims=True)
        sum_[sum_ == 0] = 1.0
        return features / sum_


    features1 = np.vstack((np.arange(n), x1)).T
    features2 = np.vstack((np.arange(n), x2)).T

    # NORMALIZATION
    factor = np.sum(np.arange(n - 1))
    features1 = compute_normalized_features(features1) * factor
    features2 = compute_normalized_features(features2) * factor
    # weigh index vs binding affinity
    features1[:, 0] *= 0.99
    features1[:, 1] *= (1 - 0.99)
    features2[:, 0] *= 0.99
    features2[:, 1] *= (1 - 0.99)

    features1 = diffuse_signal(adj1, features1, tau)
    features2 = diffuse_signal(adj2, features2, tau)

    M = ot.dist(features1, features2, metric="euclidean")
    p_single = np.full(n, 1.0 / n)
    q_single = np.full(n, 1.0 / n)

    # Compute Wasserstein distance
    return ot.emd2(p_single, q_single, M)


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
            self.adjacency_matrices is a list of M sparse COO matrices that are (N, N) 
            self.node_weights is shape (M, N)
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


    def fgw_distance_batch(self, g1, g2, x1, x2, alpha, cost, gvmax=100.0,
                           scale_cost='unitless', mass=1.0, num_cores=1):
        """
        Computes the FGW on batches of attributed graphs
        Specifically, we batch across the first dimension (M):
            g1: MxNxN
            x1: MxN

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
        window_weights = max_graph / global_max_graph + o_max_graph / global_max_o_graph

        common_max = np.maximum(max_graph, o_max_graph)

        # Condition 1: both graphs are zero matrices
        graphs_zero = (common_max == 0)

        # Condition 2: more than 80% of node feature bins are zero
        node_sparse_1 = (np.sum(x1 == 0, axis=1) / x1.shape[1]) > 0.8
        node_sparse_2 = (np.sum(x2 == 0, axis=1) / x2.shape[1]) > 0.8
        node_sparse = node_sparse_1 | node_sparse_2

        # Condition 3: both graphs after percentile clipping are zero matrices
        g1_flat = g1.reshape(g1.shape[0], -1)
        g2_flat = g2.reshape(g2.shape[0], -1)
        g1_vmax = np.percentile(g1_flat, gvmax, axis=1)
        g2_vmax = np.percentile(g2_flat, gvmax, axis=1)
        graphs_percentile_zero = (g1_vmax == 0) & (g2_vmax == 0)

        skip_window = graphs_zero | node_sparse | graphs_percentile_zero

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
        M_single = M_single.astype(np.float32)

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
                                tol=1e-6,
                                numItermax=1000,
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
                                tol_abs=1e-6,
                                max_iter=1000,
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
                            tol=1e-6,
                            numItermax=1000,
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
                            tol_abs=1e-6,
                            max_iter=1000,
                        )
                    distances.append(dist)
            distances = np.array(distances)

        return distances, window_weights
    
    def diffusion_wasserstein_batch(self, o_chrom, tau, num_cores=1):
        """
        Computes the Diffusion Wasserstein distance on batches of graphs.
        Uses sparse matrices directly for efficiency with expm_multiply.
        
        Parameters
        ----------
        o_chrom : ChromBinData
            The other chromosome to compare against
        tau : float
            Diffusion time parameter
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

        # Determine which windows to skip (basic sparsity check)
        # We can reuse the logic from fgw_distance_batch or do a simpler check here
        # For now, let's skip if node weights are mostly zero
        # node_sparse_1 = (np.sum(weights1 == 0, axis=1) / N) > 0.8
        # node_sparse_2 = (np.sum(weights2 == 0, axis=1) / N) > 0.8
        # skip_window = node_sparse_1 | node_sparse_2

        tasks = [
            (adj_list1[i], adj_list2[i], weights1[i], weights2[i], tau, False)
            for i in range(M)
        ]

        if num_cores > 1:
            with mp.Pool(processes=num_cores) as pool:
                distances = np.array(pool.map(_diffusion_wasserstein_worker, tasks))
        else:
            distances = np.array([_diffusion_wasserstein_worker(t) for t in tasks])

        # Calculate window weights for aggregation (similar to FGW)
        # Since we don't have dense matrices here easily, we can approximate or compute max from sparse
        # For simplicity, let's use max degree or max weight as a proxy if needed, 
        # but usually this is returned for weighted averaging of the final score.
        # Let's compute max element of adjacency for consistency with FGW logic if possible,
        # or just return ones if not critical. 
        # To match FGW return signature, we might want to compute it.
        
        max_vals1 = np.array([m.max() if m.nnz > 0 else 0 for m in adj_list1])
        max_vals2 = np.array([m.max() if m.nnz > 0 else 0 for m in adj_list2])
        
        global_max1 = np.max(max_vals1) if len(max_vals1) > 0 else 1.0
        global_max2 = np.max(max_vals2) if len(max_vals2) > 0 else 1.0
        
        # Avoid div by zero
        global_max1 = global_max1 if global_max1 > 0 else 1.0
        global_max2 = global_max2 if global_max2 > 0 else 1.0

        window_weights = (max_vals1 / global_max1) + (max_vals2 / global_max2)

        return distances, window_weights
    
    def wasserstein_batch(self, o_chrom, feat, weight, num_cores=1):
        """
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



    def compare(
        self,
        o_chrom: 'ChromBinData',
        alpha: float,
        method: str,
        cost : str,
        output_dir: str = 'output',
        do_output_graph: bool = False,
        param_str: str = '',
        gvmax: float = 100.0,
        scale_cost: str = 'unitless',
        mass: float = 1.0,
        feat: str = 'index_BA',
        weight: str = 'uniform',
        num_cores: int = 1
    ):
        """
        Compares this chromosome with another chromosome using the specified method

        The sparse matrices are all converted to dense matrices for batch comparison.
        Specifically:
        1. Adjacency matrices: MxNxN
        2. Node weights: MxN
        where M is the number of windows and N is the number of bins per window
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

        elif method == "dw":

            distances, window_weights = self.diffusion_wasserstein_batch(o_chrom, tau=5e-6, 
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

        return result









    
        
        

