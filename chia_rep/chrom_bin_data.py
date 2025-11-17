import sys
from math import ceil
import numpy as np
import scipy.stats as sp
from scipy.linalg import expm
from scipy.spatial.distance import cosine, jensenshannon
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


def output_graph(g, n, parent_dir, chrom, sample_name, do_output=False):
    """
    Outputs the graphs as images for visual inspection
    """
    if not do_output:
        return
    
    if g is None and n is None:
        return

    if chrom == "chr1":
        # Only save if chromosome 1
        np.save(f'{parent_dir}/{sample_name}_{chrom}_graph.npy', g)
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
    C1_i, C2_i, p_i, q_i, alpha, M_single, skip_window, N = args
    if skip_window:
        return np.nan
    
    # Unitless scaling
    # Need to compute W and GW
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

    return ot.gromov.fused_gromov_wasserstein2(
        M=M_single,
        C1=C1_i * np.sqrt(w / gw),
        C2=C2_i * np.sqrt(w / gw),
        p=p_i,
        q=q_i,
        alpha=alpha,
        loss_fun="square_loss",
        symmetric=True,
        tol_abs=1e-6,
        max_iter=1000,
    )


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

                
    def fgw_distance_batch(self, g1, g2, x1, x2, alpha, cost, max_graph, o_max_graph, gvmax=99.5, num_cores=1):
        """
        Computes the FGW on batches of attributed graphs
        Specifically, we batch across the first dimension (M):
            g1: MxNxN
            x1: MxN

        Returns a list of FGW distances for each pair in the batch and the window weights
        """
        # max_graph = np.max(g1, axis=(1, 2))
        # o_max_graph = np.max(g2, axis=(1, 2))

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

        if cost == "linear_pairs":
            # Deprecated but here to compare with previous results
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
                (C1[i], C2[i], p[i], q[i], alpha, M_single, skip_window[i], N)
                for i in range(M)
            ]
            with mp.Pool(processes=num_cores) as pool:
                distances = np.array(pool.map(_fgw_distance_worker, tasks))
        else:
            # Fallback to single core 
            distances = []
            for i in range(M):
                if skip_window[i]:
                    distances.append(np.nan)
                else:

                    # Unitless scaling
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

                    dist = ot.gromov.fused_gromov_wasserstein2(
                        M=M_single,
                        C1=C1[i] * np.sqrt(w / gw),
                        C2=C2[i] * np.sqrt(w / gw),
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
    


    def compare(
        self,
        o_chrom: 'ChromBinData',
        alpha: float,
        method: str,
        cost : str,
        output_dir: str = 'output',
        do_output_graph: bool = False,
        param_str: str = '',
        gvmax: float = 99.5,
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
        # Precompute the max values for window weights and if cost='linear'
        # Remove this if we don't need this later on
        max_graph = np.array([mat.max() if mat.nnz > 0 else 0 for mat in self.adjacency_matrices])
        o_max_graph = np.array([mat.max() if mat.nnz > 0 else 0 for mat in o_chrom.adjacency_matrices])

        adjacency_matrices = np.array([mat.toarray() for mat in self.adjacency_matrices])
        node_weights = self.node_weights # already dense

        o_adjacency_matrices = np.array([mat.toarray() for mat in o_chrom.adjacency_matrices])
        o_node_weights = o_chrom.node_weights # already dense

        if adjacency_matrices.shape != o_adjacency_matrices.shape:
            log.error(f'Adjacency matrices shape mismatch between {self.sample_name} {self.name} '
                      f'and {o_chrom.sample_name} {o_chrom.name}: '
                      f'{adjacency_matrices.shape} vs {o_adjacency_matrices.shape}')
            raise ValueError(f'Adjacency matrices shape mismatch between {self.sample_name} {self.name} '
                             f'and {o_chrom.sample_name} {o_chrom.name}: '
                             f'{adjacency_matrices.shape} vs {o_adjacency_matrices.shape}')
        

        # Output graphs if chr1
        comparison_name = f'{self.sample_name}_{o_chrom.sample_name}'
        parent_dir = f'{output_dir}/{param_str}/comparisons/{comparison_name}'
        output_graph(adjacency_matrices, node_weights, parent_dir, self.name, self.sample_name, do_output_graph)
        output_graph(o_adjacency_matrices, o_node_weights, parent_dir, o_chrom.name, o_chrom.sample_name, do_output_graph)

        if method == "fgw":
            distances, window_weights = self.fgw_distance_batch(g1=adjacency_matrices,
                                            g2=o_adjacency_matrices,
                                            x1=node_weights,
                                            x2=o_node_weights,
                                            alpha=alpha,
                                            cost=cost,
                                            max_graph=max_graph,
                                            o_max_graph=o_max_graph,
                                            gvmax=gvmax,
                                            num_cores=num_cores,
                                            )
            result = (distances, window_weights, max_graph, o_max_graph)

        elif method == "gjsd":
            # Not yet implemented
            result = None
            pass

        
        return result









    
        
        

