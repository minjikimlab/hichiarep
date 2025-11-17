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

from .util import *

log = logging.getLogger()
log_bin = logging.getLogger('bin')

MAX_USHRT = 65535

PEAK_START_INDEX = 0
PEAK_END_INDEX = 1
PEAK_LEN_INDEX = 2
PEAK_MAX_VALUE_INDEX = 3
PEAK_MEAN_VALUE_INDEX = 4

# The length of each normalization call
NORM_LEN = 100

def get_features(window_start, window_end, bin_size, node_features):
    """
    Get the node features for this window

    Parameters
    ----------
    window_start : int
        Genomic coordinates of window start
    window_end : int
        Genomic coordinates of window end
    bin_size : int
        Size of the bins
    node_features : np.ndarray
        The 1D node features for the entire chromosome

    Returns
    -------
    np.ndarray
        The indexed features for the current window
    """
    start_idx = np.floor(window_start / bin_size).astype(int)
    end_idx = np.ceil(window_end / bin_size).astype(int)
    return node_features[start_idx:end_idx]


def cosine_distance(x1, x2):
    """
    A baseline distance metric using cosine distance between the node features
    The idea is that we should see better separation than this simple baseline by
    incorporating *additional* information using structure
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Node features
        
    Returns
    -------
    float
        Cosine distance between the two feature vectors
    """
    x1_sum = x1.sum()
    x2_sum = x2.sum()

    if x1_sum == 0 and x2_sum == 0:
        # If both are zero vectors, then return 0 distance (i.e. same)
        return 0
    if x1_sum == 0 or x2_sum == 0:
        # If one is a zero vector, return max distance (i.e. different)
        return 1
    
    return cosine(x1, x2) # Already a distance (scipy.spatial.distance)


def ot_distance(x1, x2):
    """
    A baseline distance metric using optimal transport distance between the node features

    Note that implementation is identical to `emd` function, but kept separate for clarity
    We also do not return the weight here
    """

    x1_sum = x1.sum()
    x2_sum = x2.sum()

    if x1_sum == 0 and x2_sum == 0:
        # If both are zero vectors, then return 0 distance (i.e. same)
        return 0
    if x1_sum == 0 or x2_sum == 0:
        # If one is a zero vector, return max distance (i.e. different)
        return x1.size
    
    p = x1 / x1_sum
    q = x2 / x2_sum
    return c_emd(p, q, p.size) 

def center_with_mu(A, mu):
    """
    Helper function for `whiten_matrix`
    """
    return A - mu

def covariance_with_mu(A_for_corr):
    """
    Helper function for `whiten_matrix`
    """
    mu = np.mean(A_for_corr, axis=0, keepdims=True)
    X = A_for_corr - mu
    n = X.shape[0]
    C = (X.T @ X) / (n - 1)
    return C, mu


def whiten_matrix(A_for_corr, A_for_whiten, epsilon):
    """
    Whitening scheme 

    Parameters
    ----------
    A_for_corr : numpy.ndarray
        The matrix for which to compute the covariance or correlation matrix
        This is used to compute the covariance matrix C
    A_for_whiten : numpy.ndarray
        The matrix to be whitened
        This is used to apply the whitening transformation
    epsilon : float
        A small value to prevent division by zero during whitening
    
    Returns
    -------
    numpy.ndarray
        The whitened matrix
    """
    C, mu = covariance_with_mu(A_for_corr)

    d, V = np.linalg.eigh(C)

    # Truncate in case of noise
    keep = d > epsilon
    if not keep.any():
        W = np.eye(C.shape[0])
    else:
        W = (V[:, keep] * (1.0 / np.sqrt(d[keep]))) @ V[:, keep].T

    Z = center_with_mu(A_for_whiten, mu)

    if Z.ndim == 2:
        Z = Z.squeeze()

    return Z @ W, C

def whitened_cosine_distance(g1, g2, x1, x2):
    """
    Whitens the feature vectors using the covariance matrix from the graph structures
    then computes the cosine distance between the whitened vectors

    This should be akin to the "Mahalonobis distance" but using cosine distance instead of Euclidean
    """
    g1_sym = g1.T + g1 - np.diag(np.diag(g1))
    g2_sym = g2.T + g2 - np.diag(np.diag(g2))

    max_graph = np.max(g1_sym)
    max_o_graph = np.max(g2_sym)

    if max_graph == 0 and max_o_graph == 0:
        # Specializes to cosine distance between x1 and x2
        return cosine(x1, x2)
    elif max_graph == 0:
        g1_sym = np.eye(g1.shape[0])
    elif max_o_graph == 0:
        g2_sym = np.eye(g2.shape[0])

    # Whiten the feature vectors
    x1_whitened, _ = whiten_matrix(g1_sym, x1, epsilon=1e-3)
    x2_whitened, _ = whiten_matrix(g2_sym, x2, epsilon=1e-3)

    # Compute the cosine distance
    return cosine(x1_whitened, x2_whitened)


def normalized_laplacian(A):
    """
    Computes the normalized Laplacian of adjacency matrix A
    """
    if np.allclose(A, np.eye(A.shape[0])):
        return A

    d = A.sum(axis=1)
    d[np.isclose(d, 0)] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

from scipy.stats import entropy

def transform_spectra(eigs, density_method, tol=1e-3):
    """
    Helper function for `qjsd_efficient`
    Transforms the eigenvalues based on the specified density method
    """
    # Do expect the eigenvalues to be non-negative, but we 
    # do this explicitly in case of numerical instability
    eigs_nn = np.clip(eigs, a_min=0, a_max=None)

    if density_method == "L":
        return eigs_nn
    elif isinstance(density_method, str) and density_method[0] == "L":
        return eigs_nn ** int(density_method[1:])
    elif density_method == "+":
        mask = eigs_nn > tol
        inv_sqrt = np.zeros_like(eigs_nn)
        inv_sqrt[mask] = 1 / np.sqrt(eigs_nn[mask])
        return inv_sqrt
    elif isinstance(density_method, (float, int)) and density_method > 0:
        return np.exp(-density_method * eigs_nn)
    else:
        raise ValueError(f"Unknown density_method: {density_method}")
    
def von_neumann_entropy_from_eigs(eigs, base=2):
    eigs_nn = np.clip(eigs, a_min=0, a_max=None)
    p = eigs_nn / np.sum(eigs_nn)
    return entropy(p, base=base)
    

def qjsd_efficient(A1, A2, base=2, density_method="L"):
    """
    Computes the quantum Jensen-Shannon divergence between two graphs
    """
    L1 = normalized_laplacian(A1)
    L2 = normalized_laplacian(A2)

    # Assume L1 and L2 are symmetric
    eigs1, U1 = np.linalg.eigh(L1)
    eigs2, U2 = np.linalg.eigh(L2)

    eigs_trans_1 = transform_spectra(eigs1, density_method)
    eigs_trans_2 = transform_spectra(eigs2, density_method)

    # Make more efficient
    rho1 = U1 @ np.diag(eigs_trans_1) @ U1.T
    rho2 = U2 @ np.diag(eigs_trans_2) @ U2.T
    rho_m = 0.5 * (rho1 / np.trace(rho1) + rho2 / np.trace(rho2))
    eigs_trans_m = np.linalg.eigvalsh(rho_m)

    H_vn_A1 = von_neumann_entropy_from_eigs(eigs_trans_1, base=base)
    H_vn_A2 = von_neumann_entropy_from_eigs(eigs_trans_2, base=base)
    H_vn_M = von_neumann_entropy_from_eigs(eigs_trans_m, base=base)

    return H_vn_M - 0.5 * (H_vn_A1 + H_vn_A2)


def graph_jsd(g1, g2, x1, x2, alpha, density_method):
    """
    Convex combination of classical JSD and quantum JSD
    comparing binding affinity and 3D structure, respectively

    Parameters
    ----------
    density_method : str or float
        Method to construct the density matrix from adjacency matrix
        Options: 
            'L' for Laplacian
            'L2' for Laplacian squared
            Positive float for t of e^(-t*L)
            '+' for sqrt pseudoinverse
    """

    g1_sym = g1.T + g1 - np.diag(np.diag(g1))
    g2_sym = g2.T + g2 - np.diag(np.diag(g2))

    max_graph = np.max(g1_sym)
    max_o_graph = np.max(g2_sym)
    
    # If both graphs are empty, they are the same graph
    # We should still compare the features
    if max_graph == 0 and max_o_graph == 0:
        alpha = 1.0  # only consider classical JSD
    elif max_graph == 0:
        g1_sym = np.eye(g1_sym.shape[0])
    elif max_o_graph == 0:
        g2_sym = np.eye(g2_sym.shape[0])

    p = x1 / np.sum(x1) if np.sum(x1) > 0 else np.ones_like(x1) / len(x1)
    q = x2 / np.sum(x2) if np.sum(x2) > 0 else np.ones_like(x2) / len(x2)

    if np.isclose(alpha, 1):
        c_jsd = jensenshannon(p, q, base=2) ** 2
        q_jsd = 0
    elif np.isclose(alpha, 0):
        c_jsd = 0
        q_jsd = qjsd_efficient(g1_sym, g2_sym, base=2, density_method=density_method)
    else:
        c_jsd = jensenshannon(p, q, base=2) ** 2
        q_jsd = qjsd_efficient(g1_sym, g2_sym, base=2, density_method=density_method)

    combined_jsd = alpha * c_jsd + (1 - alpha) * q_jsd

    return combined_jsd

def diffusion_distance(A, t):
    """
    Computes the heat kernel e^(-tL) and returns the diffusion distance matrix
    Helper for `fgw_distance`
    """
    L = normalized_laplacian(A)
    K_t = expm(-t * L)
    diag = np.diag(K_t)
    D2 = diag[:, None] + diag[None, :] - 2 * K_t
    np.maximum(D2, 0, out=D2) # for numerical stability 
    return np.sqrt(D2, dtype=np.float32)

def flip(A, common_max):
    """ 
    Compute the element-wise flip
    """
    A_norm = A / common_max
    return  1.0 - A_norm


def fgw_distance(g1, g2, x1, x2, use_node_features_as_weights, alpha=0.5):
    """
    The bounded FGW distance. We ensure that the feature-feature cost matrix and the 
    structure-structure cost matrix are both bounded within [0, 1], which should ensure
    the FGW to be bounded within [0, 1] as well

    There are two options to incorporate the node features
    1. Use the node features as the features of FGW
    2. Use the node features as weights i.e. PMF of the distributions, and the
        linear distance dependent decay function as the features of FGW

    Parameters
    ----------
    g1, g2 : np.ndarray
        2D graph adjacency matrices
    x1, x2 : np.ndarray
        Node features
    use_node_features_as_weights : bool
        If True, use node features as FGW weights and linear distance decay as FGW features
        Otherwise, use uniform PMF as FGW weights and node features as FGW features
    alpha : float
        Trade-off parameter

    Returns
    -------
    float
        The bounded FGW distance
    """    
    g1_sym = g1.T + g1 - np.diag(np.diag(g1))
    g2_sym = g2.T + g2 - np.diag(np.diag(g2))

    max_graph = np.max(g1_sym)
    max_o_graph = np.max(g2_sym)

    if max_graph == 0 and max_o_graph == 0:
        # Specializes to just OT on the features
        return c_emd(x1, x2, x1.size)
    elif max_graph == 0:
        g1_sym = np.eye(g1_sym.shape[0])
    elif max_o_graph == 0:
        g2_sym = np.eye(g2_sym.shape[0])

    # Options for structure-structure cost matrix
    # C1 = shortest_path(g1_sym, directed=False, unweighted=False)
    # C2 = shortest_path(g2_sym, directed=False, unweighted=False)

    # C1 = diffusion_distance(g1_sym, t=2)
    # C2 = diffusion_distance(g2_sym, t=2)

    common_max = max(max_graph, max_o_graph)
    C1 = flip(g1_sym, common_max)
    C2 = flip(g2_sym, common_max)

    if use_node_features_as_weights:
        # Use node features as weights (PMF)
        p = x1 / x1.sum() if x1.sum() > 0 else np.ones_like(x1) / x1.size
        q = x2 / x2.sum() if x2.sum() > 0 else np.ones_like(x2) / x2.size
        
        # Create linear distance decay feature matrix
        coords1 = np.arange(g1.shape[0]).reshape(-1, 1)
        coords2 = np.arange(g2.shape[0]).reshape(-1, 1)

        M = ot.dist(coords1, coords2, metric='euclidean')
    else:
        # Use uniform weights (PMF)
        p = np.ones(g1.shape[0]) / g1.shape[0]
        q = np.ones(g2.shape[0]) / g2.shape[0]
        
        # Use node features as FGW features
        if x1.sum() > 0:
            x1_normalized = x1 / x1.sum()
        else:
            x1_normalized = np.ones_like(x1) / x1.size
        
        if x2.sum() > 0:
            x2_normalized = x2 / x2.sum()
        else:
            x2_normalized = np.ones_like(x2) / x2.size
        
        M = ot.dist(x1_normalized.reshape(-1, 1), x2_normalized.reshape(-1, 1), metric='euclidean')

    # Compute FGW distance
    fgw_dist = ot.gromov.fused_gromov_wasserstein2(M=M, C1=C1, C2=C2, p=p, q=q, alpha=alpha, loss_fun='square_loss')

    return fgw_dist


def emd(
    p: np.ndarray,
    q: np.ndarray
) -> Tuple[float, float]:
    """
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


def output_graph(
    output_dir: str,
    chrom_name: str,
    window_start: int,
    window_end: int,
    graph: np.ndarray,
    sample_name: str,
    data_type: str,
    do_output: bool = False
) -> None:
    """
    Outputs graph to file as both .txt and .npy

    Parameters
    ----------
    output_dir
        Directory to save files
    chrom_name
        Name of the chromosome
    window_start
        Start coordinate of the window
    graph
        The numpy array to save
    sample_name
        Name of the sample
    data_type
        A string to identify the type of data (e.g., 'emd_graph', 'fgw_features')
    do_output
        Flag to enable or disable saving

    Returns
    -------
    None
    """
    if not do_output or graph is None:
        return
    
    if chrom_name != "chr1":
        # Only generate statistics for chr1 to not overwhelm the output folder
        return

    start_time = time.time()
    
    # Define base path 
    base_path = f'{output_dir}/{sample_name}_{chrom_name}_{window_start}_{data_type}'

    # Save .npy file
    np.save(f'{base_path}.npy', graph)

    # Save summary and flattened graph to .txt file
    with open(f'{base_path}.txt', 'w') as out_file, \
            np.printoptions(threshold=sys.maxsize, precision=6,
                            linewidth=sys.maxsize, suppress=True) as _:
        n_non_zeros = np.count_nonzero(graph)
        n_zeros = graph.size - n_non_zeros
        
        header = (f'sample:{sample_name}\tchrom:{chrom_name}\t'
                  f'window_start:{window_start}\tdata_type:{data_type}\t'
                  f'non_zeros:{n_non_zeros}\tzeros:{n_zeros}\n')
        out_file.write(header)

        if graph.ndim > 1:
            # flat_graph = '\t'.join([str(x) for x in graph.flatten()])
            out_file.write(str(graph)) # Print as matrix
        else:
            flat_array = '\t'.join([str(x) for x in graph])
            out_file.write(flat_array)

    log.debug(f'Saving {data_type} for {sample_name} took {time.time() - start_time}s')


class ChromLoopData:
    """
    A class used to represent a chromosome in a sample

    Attributes
    ----------
    name : str
        Name of chromosome
    size : int
        Size of chromosome
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    """

    def __init__(
        self,
        chrom_name: str,
        chrom_size: int,
        sample_name: str
    ):
        """
        Parameters
        ----------
        chrom_name : str
            Name of chromosome
        chrom_size : int
            Size of chromosome
        sample_name : str
            Name of the sample (LHH0061, LHH0061_0061H, ...)
        """

        self.name = chrom_name
        self.size = chrom_size
        self.sample_name = sample_name

        self.start_anchor_list = [[], []]
        self.end_anchor_list = [[], []]
        self.start_list = None  # Later initialized with bedgraph
        self.end_list = None
        self.value_list = []
        self.pet_count_list = []
        self.numb_loops = 0
        self.removed_intervals = [[], []]  # Start/end anchors are on same peak
        self.start_list_peaks = None
        self.end_list_peaks = None

        self.norm_list = None

        self.filtered_start = []
        self.filtered_end = []
        self.filtered_values = []
        self.filtered_numb_values = 0
        self.filtered_anchors = []
        self.kept_indexes = []

        # Used in filter_with_peaks, keeps track of peaks for each loop
        self.peak_indexes = []
        self.peaks_used = None

        self.max_loop_value = 0

        # Attributes for FGW
        self.filtered_pet_count_list = []  # Populated in ChromLoopData preprocess by filter_with_peaks
        self.node_features = None  # Populated in GenomeLoopData read in phase

    def add_loop(
        self,
        loop_start1: int,
        loop_end1: int,
        loop_start2: int,
        loop_end2: int,
        loop_value: int
    ) -> None:

        self.start_anchor_list[0].append(loop_start1)
        self.start_anchor_list[1].append(loop_end1)
        self.end_anchor_list[0].append(loop_start2)
        self.end_anchor_list[1].append(loop_end2)

        self.value_list.append(loop_value) # Initialized to PET counts before being overwritten 
        # in (finish_init -> find_loop_anchor_points) to weighted values

        self.numb_loops += 1

    def finish_init(
        self,
        bedgraph: BedGraph
    ) -> bool:
        """
        Finishes the construction of this chromosome. Converts lists to numpy
        arrays and calls find_loop_anchor_points

        Parameters
        ----------
        bedgraph : BedGraph
            Used to find the anchor points of each loop

        Returns
        -------
        bool
            Whether the chromosome was successfully made
        """

        if self.numb_loops == 0:
            return False

        self.pet_count_list = np.asarray(self.value_list, dtype=np.uint32)
        self.value_list = np.asarray(self.value_list, dtype=np.float64)
        self.start_anchor_list = np.asarray(self.start_anchor_list,
                                            dtype=np.int32)
        self.end_anchor_list = np.asarray(self.end_anchor_list, dtype=np.int32)

        log.debug(f"Max PET count: {np.max(self.pet_count_list)}")

        # if not bedgraph.has_chrom(self.name):
        if self.name not in bedgraph.chromosome_map:
            log.warning(f"{self.name} was not found in corresponding bedgraph: "
                        f"{bedgraph.name}")
            return False

        self.find_loop_anchor_points(bedgraph)
        return True

    def find_loop_anchor_points(
        self,
        bedgraph: BedGraph
    ):
        """
        Finds the exact loop anchor points.

        Finds peak values for each anchor and weighs the loop. Also finds loops
        that have overlapping start/end indexes due to close and long start/end
        anchors.

        Parameters
        ----------
        bedgraph : BedGraph
            Used to find the anchor points of each loop
        """

        log.info(f'Finding anchor points for {self.sample_name}\'s {self.name}'
                 f' from {bedgraph.name}')

        bedgraph.load_chrom_data(self.name)

        # Get index of peaks in every anchor interval
        self.start_list = bedgraph.stats(start_list=self.start_anchor_list[0],
                                         end_list=self.start_anchor_list[1],
                                         chrom_name=self.name, stat='max_index')
        self.end_list = bedgraph.stats(start_list=self.end_anchor_list[0],
                                       end_list=self.end_anchor_list[1],
                                       chrom_name=self.name, stat='max_index')

        # Get peak value for every anchor interval
        # Note that the "peak" here is NOT referring to the peak called output but rather
        # just referring to the binding affinity signal as a peak
         # The above extracts the stat='max_index' but this extracts the signal value itself: stat='max'
        start_list_peaks = bedgraph.stats(start_list=self.start_anchor_list[0], 
                                          end_list=self.start_anchor_list[1],
                                          chrom_name=self.name, stat='max')
        end_list_peaks = bedgraph.stats(start_list=self.end_anchor_list[0],
                                        end_list=self.end_anchor_list[1],
                                        chrom_name=self.name, stat='max')
        self.start_list_peaks = start_list_peaks
        self.end_list_peaks = end_list_peaks
        bedgraph.free_chrom_data(self.name)

        start_list_peaks = start_list_peaks / start_list_peaks.sum()
        end_list_peaks = end_list_peaks / end_list_peaks.sum()

        for i in range(self.numb_loops):
            # loop_start = self.start_list[i]
            # loop_end = self.end_list[i]

            # Remove anchors that have the same* peak
            # Keep indexes of loop length to avoid comparisons in interval
            # if not loop_start < loop_end:
            #     self.value_list[i] = 0
            #
            #     # Removed interval goes from
            #     # (start of start anchor, end of end anchor)
            #     self.removed_intervals[0].append(self.start_anchor_list[0][i])
            #     self.removed_intervals[1].append(self.end_anchor_list[1][i])
            #     continue

            # Weigh each loop based on its corresponding bedgraph peak
            # peak_value = max(start_list_peaks[i], end_list_peaks[i])
            peak_value = start_list_peaks[i] + end_list_peaks[i]
            self.value_list[i] *= peak_value

        self.max_loop_value = np.max(self.value_list)

        # Should be very small due to peaks being weighted earlier
        log.debug(f"Max loop weighted value: {self.max_loop_value}")

    # May have more pre-processing to do besides filtering later?
    # Useless extra function otherwise
    def preprocess(
        self,
        peak_list: list,
        both_peak_support: bool = False
    ) -> bool:
        """
        DEPRECATED
        """
        return self.filter_with_peaks(peak_list, both_peak_support)

    def filter_with_peaks(
        self,
        peak_list: list,
        both_peak_support: bool = False
    ) -> bool:
        """
        DEPRECATED

        Filters out loops without peak support.

        Get coverage of peaks that have been chosen to be used. Find loops that
        are not within that coverage and filter them out.

        Parameters
        ----------
        peak_list : list(list)
            List of peaks to use
        both_peak_support : bool, optional
            Whether to only keep loops that have peak support on both sides
            (default is False)

        Returns
        -------
        bool
            Whether the chromosome had any problems when filtering
        """

        start_time = time.time()

        num_peaks = len(peak_list)
        min_peak_value = peak_list[-1][PEAK_MAX_VALUE_INDEX]

        log.info(f"Filtering {self.sample_name} {self.name} with "
                 f"{num_peaks} peaks...")
        log.debug(f"Top peaks: {peak_list[:3]}")
        log.debug(f"Bottom peaks: {peak_list[-3:]}")
        log.debug(f'Min peak value: {min_peak_value}')

        # Get the coverage of each wanted peak
        # Could be used to find the specific peaks for every loop
        index_array = np.zeros(self.size, dtype=np.uint16)

        if num_peaks >= MAX_USHRT:
            log.warning(f'Number of peaks: {num_peaks} is greater than max_unsigned_short: {MAX_USHRT}')
        for i in range(num_peaks):
            peak_start = peak_list[i][0]
            peak_end = peak_list[i][1]
            index_array[peak_start:peak_end] = i + 1

        log.debug(f'Time: {time.time() - start_time}')

        numb_deleted = 0
        removed_loop_lengths = []
        removed_loop_values = []
        kept_loop_lengths = []
        self.kept_indexes = []
        self.peak_indexes = [[], []]
        self.filtered_start = []
        self.filtered_end = []
        self.filtered_values = []
        self.filtered_anchors = []
        self.filtered_pet_count_list = []

        for i in range(self.numb_loops):
            loop_start = self.start_list[i]
            loop_end = self.end_list[i]
            loop_value = self.value_list[i]

            if loop_start > loop_end:
                temp_val = loop_start
                loop_start = loop_end
                loop_end = temp_val

            if loop_value == 0:
                continue

            if both_peak_support:
                to_keep = index_array[loop_start] and index_array[loop_end]
            else:
                to_keep = index_array[loop_start] or index_array[loop_end]

            if not to_keep:
                removed_loop_values.append(loop_value)
                numb_deleted += 1
                removed_loop_lengths.append(loop_end - loop_start)
                continue

            self.filtered_start.append(loop_start)
            self.filtered_end.append(loop_end)
            self.filtered_values.append(loop_value)
            self.filtered_pet_count_list.append(self.pet_count_list[i])  # Have a filtered version with non-weighted raw adjacency matrix PET counts
            self.filtered_anchors.append([self.start_anchor_list[0][i],
                                          self.start_anchor_list[1][i],
                                          self.start_list_peaks[i]])
            self.filtered_anchors.append([self.end_anchor_list[0][i],
                                          self.end_anchor_list[1][i],
                                          self.start_list_peaks[i]])
            self.kept_indexes.append(i)

            kept_loop_lengths.append(loop_end - loop_start)

            # Unused for now
            self.peak_indexes[0].append((
                peak_list[index_array[loop_start] - 1][0],
                peak_list[index_array[loop_start] - 1][1]))
            self.peak_indexes[1].append((
                peak_list[index_array[loop_end] - 1][0],
                peak_list[index_array[loop_end] - 1][1]))

        self.filtered_start = np.array(self.filtered_start, dtype=np.int32)
        self.filtered_end = np.array(self.filtered_end, dtype=np.int32)
        self.filtered_values = np.array(self.filtered_values)
        self.filtered_pet_count_list = np.array(self.filtered_pet_count_list, dtype=np.uint32)
        self.filtered_numb_values = self.filtered_start.size
        self.kept_indexes = np.array(self.kept_indexes, dtype=np.int32)

        log.debug(f'Total loops: {self.numb_loops}')
        log.debug(f"Number of loops removed: {numb_deleted}")
        log.info(f"Number of loops kept: {self.filtered_numb_values}")

        if self.filtered_numb_values == 0:
            log.warning(f"No loops left. Skipping")
            return False

        if numb_deleted > 0:
            log.debug(
                f'Avg loop length removed: {np.mean(removed_loop_lengths)}')
            log.debug(f'Avg loop value removed: {np.mean(removed_loop_values)}')
        else:
            log.debug(f'Avg loop length removed: N/A')
            log.debug(f'Avg loop value removed: N/A')
        log.debug(f'Avg loop length kept: {np.mean(kept_loop_lengths)}')
        log.debug(f'Avg loop value kept: {np.mean(self.filtered_values)}')
        log.debug(f'Largest loop value kept: {np.max(self.filtered_values)}')
        log.debug(f'Time taken: {time.time() - start_time}\n')

        return True

    def create_graph(
        self,
        loop_starts: np.ndarray,
        loop_ends: np.ndarray,
        loop_values: np.ndarray,
        bin_size: int,
        window_start: int,
        window_size: int,
        random: bool = False,
        num_loops: int = 0,
    ) -> np.ndarray:
        """
        Creates a bin-based graph to easily compare loops

        Parameters
        ----------
        loop_starts : np.ndarray
            Start positions of loops
        loop_ends : np.ndarray
            End positions of loops
        loop_values : np.ndarray
            Values of loops
        bin_size : int
        window_start : int
            Added to compute the proper loop start and end index with respect to the 
            adjacency matrix indices
        window_size : int
        random : bool, optional
            Randomly pick which loops to use (Default is False)
            Useless for now since different sequencing depth is ignored
        num_loops : int, optional
            Number of loops to use when making the graph
        Returns
        -------
        np.ndarray
        """
        if not np.allclose([len(loop_starts), len(loop_ends)], len(loop_values)):
            log.error("Loop starts, ends, and values are not the same length")
            return None

        graph_len = ceil(window_size / bin_size)
        graph = np.zeros((graph_len, graph_len), dtype=np.float64)

        # So comparing a sample with no loops vs. a sample with loops doesn't
        # result in high reproducibility
        if num_loops == 0:
            num_loops = len(loop_starts)
        indexes = None
        if random:
            indexes = np.random.choice(len(loop_starts), num_loops, replace=False)

        num_loops_used = 0
        for loop_index in range(num_loops):

            start = loop_starts[loop_index]
            end = loop_ends[loop_index]
            value = loop_values[loop_index]

            orig_start = start
            orig_end = end

            # Convert genomic coordinates to graph coordinates (not yet binned)
            start = start - window_start
            end = end - window_start

            # Bin (clipping is new addition)
            bin_start = np.clip(int(start / bin_size), 0, graph_len - 1) 
            bin_end = np.clip(int(end / bin_size), 0, graph_len - 1)

            if bin_end < bin_start:
                log.error(
                    f'{orig_start}\t{orig_end}\t{start}\t{end}\t{bin_start}\t{bin_end}')
                temp_val = bin_start
                bin_start = bin_end
                bin_end = temp_val

            graph[bin_start][bin_end] += value

            # Also get areas surrounding this loop
            # May not be needed with emd calculation
            # Helps with finding jensen-shannon
            for j in range(bin_start - 1, bin_start + 2):
                if j < 0 or j == graph_len:
                    continue
                for k in range(bin_end - 1, bin_end + 2):
                    if k < 0 or k == graph_len:
                        continue
                    graph[j][k] += value

            num_loops_used += 1

        return graph
    

    def _create_option_graphs(
            self, 
            fgw: str,
            window_start: int,
            window_end: int,
            bin_size: int
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the adjacency matrix and node feature vector for the FGW distance

        Selects the appropriate loop data based on the fgw parameter

        Parameters
        ----------
        fgw : str
            The various integration methods of the Fused Gromov Wasserstein distance
            "A": Full Preprocessing
            "B": Partial Preprocessing
            "C": No Preprocessing

            Other options, like diffused, is not implemented
        window_start : int
            Added to compute the proper loop start and end index with respect to the 
            adjacency matrix indices
        window_end : int
            End of the window
        bin_size : int
            Size of the bins
        """
        eff_window_size = window_end - window_start
        graph_len = ceil(eff_window_size / bin_size)

        # Get node features for the current window
        node_features = get_features(window_start, window_end, bin_size, self.node_features)
        fgw_graph = np.zeros((graph_len, graph_len), dtype=np.float64)

        if fgw == "A":
            # Option A: Full Preprocessing 
            # FILTERED, WEIGHTED
            starts, ends, values = self.filtered_start, self.filtered_end, self.filtered_values
        elif fgw == "B1":
            # Option B1: Partial Preprocessing
            # FILTERED, RAW
            starts, ends, values = self.filtered_start, self.filtered_end, self.filtered_pet_count_list
        elif fgw == "B2":
            # Option B2: Partial Preprocessing
            # ALL, WEIGHTED
            starts, ends, values = self.start_list, self.end_list, self.value_list
        elif fgw == "C":
            # Option C: No Preprocessing
            # ALL, RAW
            starts, ends, values = self.start_list, self.end_list, self.pet_count_list
        else:
            log.error(f'FGW option {fgw} is not implemented')
            return None, None

        loops_idx = get_loops(window_start, window_end, starts, ends, values.astype(np.float64))

        window_starts = starts[loops_idx]
        window_ends = ends[loops_idx]
        window_values = values[loops_idx]

        fgw_graph = self.create_graph(window_starts, window_ends, window_values, bin_size, window_start, eff_window_size)
    
        return fgw_graph, node_features
    
        

    def get_stats(
        self,
        graph: np.ndarray,
        o_graph: np.ndarray,
        node_features: np.ndarray,
        o_node_features: np.ndarray,
        alpha: float,
        density_method: Union[str, float],
        o_chrom: 'ChromLoopData'
    ):
        """
        Get the reproducibility stat.

        Jensen Shannon and EMD. Though EMD seems to be better.

        Parameters
        ----------
        graph : np.ndarray
            2D Numpy array, float64
            Graph created from window from sample1
        o_graph : np.ndarray
            2D Numpy array, float64
            Graph created from window from sample2
        o_chrom : ChromLoopData
            sample2
            Used to get max_loop_value for the window weight calculation

        Returns
        -------
        dict
            emd_value : float
                Comparison value based on the Earth mover's Distance formula
            j_value : float
                Comparison value based on the Jensen-Shannon formula
            emd_dist : float
                Result of the Earth mover's Distance formula
            j_divergence : float
                Result of the Jensen-Shannon formula
            w : str
                Weight of this window
        """
        result = {}
        max_graph = np.max(graph)
        max_o_graph = np.max(o_graph)
        result['w'] = max_graph / self.max_loop_value + max_o_graph / o_chrom.max_loop_value

        # Disable all distance functions except FGW
        result['j_divergence'] = np.nan
        result['j_value'] = np.nan
        result['emd_value'] = np.nan
        result['linear_emd_value'] = np.nan
        result['emd_dist'] = np.nan

        # if max_graph == 0 or max_o_graph == 0:
        #     if max_graph == 0:
        #         log.debug('No loops in sample A')
        #     else:
        #         log.debug('No loops in sample B')

        #     result['j_divergence'] = 1
        #     result['j_value'] = -1
        #     result['emd_value'] = -1
        #     result['linear_emd_value'] = -1
        #     result['emd_dist'] = 1

        #     # We still need to compute the new distances
        #     # so don't return results yet

        # else:
        #     log.debug('Loops are found in both samples')

        #     graph_flat = graph.flatten()
        #     o_graph_flat = o_graph.flatten()

        #     j_divergence = jensen_shannon_divergence(graph_flat, o_graph_flat)

        #     # Make j_value range from -1 to 1
        #     j_value = 2 * (1 - j_divergence) - 1

        #     # Calculate emd for all rows and columns -> Take weighted average
        #     emd_distance_list = []
        #     emd_weight_list = []
        #     for k in range(graph[0].size):
        #         emd_dist, emd_weight = emd(graph[k], o_graph[k])
        #         emd_distance_list.append(emd_dist)
        #         emd_weight_list.append(emd_weight)

        #         emd_dist, emd_weight = emd(graph[:, k], o_graph[:, k])
        #         emd_distance_list.append(emd_dist)
        #         emd_weight_list.append(emd_weight)

        #     max_emd_weight = np.max(emd_weight_list)

        #     if max_emd_weight == 0:
        #         overall_emd_dist = 0
        #     else:
        #         overall_emd_dist = np.average(emd_distance_list,
        #                                     weights=emd_weight_list)
        #     # overall_emd_dist = np.mean(emd_distance_list)

        #     # Higher emd_dist == samples are more different
        #     # Lower emd_dist == samples are more similar
        #     max_emd_dist = graph.shape[0] - 1
        #     numerator = overall_emd_dist - max_emd_dist
        #     emd_value = 2 * numerator * numerator / (
        #             max_emd_dist * max_emd_dist) - 1

        #     # Linear scale
        #     # emd_value = 1 - 2 / max_emd_dist * overall_emd_dist
        #     # linear_emd_value = 1 - 2 * overall_emd_dist

        #     if max_emd_weight == 0 and result['w'] != 0:
        #         log.error(f'Total Weight: {result["w"]} with 0 emd dist')

        #     result['j_divergence'] = j_divergence
        #     result['j_value'] = j_value
        #     result['emd_value'] = emd_value
        #     result['emd_dist'] = overall_emd_dist
        #     # result['linear_emd_value'] = linear_emd_value


        # Disable all distance functions except FGW
        result['cosine_dist'] = np.nan
        result['white_cosine_dist'] = np.nan
        result['graph_jsd_dist'] = np.nan

        # result['cosine_dist'] = cosine_distance(node_features, o_node_features)
        # result['white_cosine_dist'] = whitened_cosine_distance(graph, o_graph, node_features, o_node_features)
        # result['graph_jsd_dist'] = graph_jsd(graph, o_graph, node_features, o_node_features, 
                                        # alpha=alpha, density_method=density_method)
        result['fgw_dist'] = fgw_distance(graph, o_graph, node_features, o_node_features, 
                                          use_node_features_as_weights=True, alpha=alpha)

        return result

    def compare(
        self,
        o_chrom: 'ChromLoopData',
        window_start: int,
        window_end: int,
        window_size: int,
        bin_size: int,
        num_peaks: any,
        option: str = None,
        alpha: float = 0.5,
        density_method: Union[str, float] = 'L2',
        output_dir: str = 'output',
        do_output_graph: bool = False
    ) -> Dict[str, float]:
        """
        Compare a window of this chromosome to another chromosome from another
        sample

        Parameters
        ----------
        o_chrom : ChromLoopData
            The other chromosome
        window_start : int
            The start of the window
        window_end : int
            The end of the window
        window_size : int
            Need this to save to a consistent output folder with parameter name
        bin_size : int
            Determines which loops are the same by putting them into bins
        num_peaks : any
        option : str
            The various processing methods of the loops to integrate new distances
            "A": Full Preprocessing
            "B1": Partial Preprocessing
            "B2": Partial Preprocessing
            "C": No Preprocessing

            Other options, like diffused, is not implemented

        output_dir : str, optional
            Directory to output data
        do_output_graph : bool, optional
            Whether to output graph used for comparison (Default is False)

        Returns
        -------
        dict
            emd_value : float
                Comparison value based on the Earth mover's Distance formula
            j_value : float
                Comparison value based on the Jensen-Shannon formula
            emd_dist : float
                Result of the Earth mover's Distance formula
            j_divergence : float
                Result of the Jensen-Shannon formula
            w : float
                Weight of this window
        """

        return_skeleton = {
            'emd_value': 0,
            'j_value': 0,
            'emd_dist': 0,
            'j_divergence': 0,
            'cosine_dist': 0,
            'white_cosine_dist': 0,
            'graph_jsd_dist': 0,
            'fgw_dist': 0,
            'w': 0,
        }

        if window_end > self.size: # chromosome size
            window_end = self.size

        if window_start >= self.size:
            log.error(f"Start of window ({window_start}) is larger than "
                      f"{self.name} size: {self.size}")
            return return_skeleton

        log.debug(f'{self.sample_name} vs. {o_chrom.sample_name} '
                  f'{self.name}:{window_start} - {window_end}')

        # Get loop indexes in the window
        # FILTERED, WEIGHTED (default)
        loops = get_loops(window_start, window_end, self.filtered_start,
                          self.filtered_end, self.filtered_values)
        o_loops = get_loops(window_start, window_end, o_chrom.filtered_start,
                            o_chrom.filtered_end, o_chrom.filtered_values)
        num_loops = len(loops)
        num_o_loops = len(o_loops)
        log.debug(f"Numb of loops in {self.sample_name}: {num_loops}")
        log.debug(f"Numb of loops in {o_chrom.sample_name}: {num_o_loops}")

        if num_loops == 0 and num_o_loops == 0:
            result = return_skeleton
            log.debug('No loops in either sample')
        else:
            # Make graphs using all loops in the window
            eff_window_size = window_end - window_start
            # graph = self.create_graph(self.filtered_start[loops], self.filtered_end[loops], self.filtered_values[loops],
            #                           bin_size, window_start, eff_window_size)
            # o_graph = o_chrom.create_graph(o_chrom.filtered_start[o_loops], o_chrom.filtered_end[o_loops], o_chrom.filtered_values[o_loops],
            #                                bin_size, window_start, eff_window_size)
            
            graph, node_features = self._create_option_graphs(option, window_start, window_end, bin_size)
            o_graph, o_node_features = o_chrom._create_option_graphs(option, window_start, window_end, bin_size)

            comparison_name = f'{self.sample_name}_{o_chrom.sample_name}'
            param_str = f'{window_size}.{bin_size}.{num_peaks}'
            parent_dir = f'{output_dir}/{param_str}/comparisons/{comparison_name}'

            # Save EMD/JSD graphs
            # output_graph(parent_dir, self.name, window_start, window_end, graph, self.sample_name, 'emd_graph', do_output=do_output_graph)
            # output_graph(parent_dir, o_chrom.name, window_start, window_end, o_graph, o_chrom.sample_name, 'emd_graph', do_output=do_output_graph)

            # Save FGW graphs and features
            output_graph(parent_dir, self.name, window_start, window_end, graph, self.sample_name, f'{option}_graph', do_output=do_output_graph)
            output_graph(parent_dir, o_chrom.name, window_start, window_end, o_graph, o_chrom.sample_name, f'{option}_graph', do_output=do_output_graph)
            output_graph(parent_dir, self.name, window_start, window_end, node_features, self.sample_name, 'node_features', do_output=do_output_graph)
            output_graph(parent_dir, o_chrom.name, window_start, window_end, o_node_features, o_chrom.sample_name, 'node_features', do_output=do_output_graph)

            result = self.get_stats(graph, o_graph, 
                                    node_features, o_node_features, 
                                    alpha, density_method,
                                    o_chrom)


        log.debug(f'emd_dist: {result["emd_dist"]}')
        log.debug(f'emd_value: {result["emd_value"]}')
        log.debug(f'j_divergence: {result["j_divergence"]}')
        log.debug(f'j_value: {result["j_value"]}')
        log.debug(f'cosine_dist: {result["cosine_dist"]}')
        log.debug(f'white_cosine_dist: {result["white_cosine_dist"]}')
        log.debug(f'graph_jsd_dist: {result["graph_jsd_dist"]}')
        log.debug(f'fgw_dist: {result["fgw_dist"]}')

        log.debug(
            '-----------------------------------------------------------------')

        return result
