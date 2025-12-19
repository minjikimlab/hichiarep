import numpy as np
import os
import logging
# import math
from pyBedGraph import BedGraph
import hicstraw
from typing import Dict, List, Union
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, FuncFormatter

from .chrom_bin_data import ChromBinData

log = logging.getLogger()

# Missing in many miseq peak files
CHROMS_TO_IGNORE = ['chrY', 'chrM']
# CHROMS_TO_IGNORE = [] # Debug

def nan_average(values, weights=None):
    masked_values = np.ma.masked_array(values, np.isnan(values))
    avg = np.ma.average(masked_values, weights=weights)
    if isinstance(avg, np.ma.MaskedArray):
        return avg.filled(np.nan)
    return avg


def check_alt_chrom_name(chrom_name, chromosome_list):
    """
    Checks for match of chrom_name in chromosome_list

    Considers alternate chromosome names (chr1 vs 1)

    Returns None if not found. Otherwise returns the matched chromosome name 

    Parameters
    ----------
    chrom_name : str
        Chromosome name to check
    chromosome_list : list
        List of chromosome names to check against
    """
    if chrom_name in chromosome_list:
        return chrom_name
    elif chrom_name.startswith('chr'):
        alt_name = chrom_name[3:]
        if alt_name in chromosome_list:
            return alt_name
    else:
        alt_name = 'chr' + chrom_name
        if alt_name in chromosome_list:
            return alt_name
    return None


def weigh_adjacency_matrix_binding_affinity(A, b):
    """
    Weighs the batch adjacency matrix (MxNxN) by the binding affinity node weights (MxN)
    according to the rule:
    A[:,i,j] = A[:,i,j] * (b[:,i] + b[:,j])

    The first dimension is the window index and is simply broadcasted over

    Parameters
    ----------
    A : np.ndarray
        MxNxN array of adjacency matrices
    b : np.ndarray
        MxN array of binding affinity node weights

    Returns
    -------
    out : np.ndarray
        MxNxN array of weighed adjacency matrices
    """
    out = A * b[:, :, None]
    out += A * b[:, None, :]
    return out

def threshold_peaks(p_array, num_peaks, total_peaks, to_remove, chrom_name, base_ratio):
    """
    Thresholds the peaks, assuming that `p_array` is a 1D numpy array of peak values

    Updates `to_remove` list for chromosomes that need to be removed

    Parameters
    ----------
    p_array : np.ndarray
        1D numpy array of peak values
    num_peaks : int
        Used by this function to simply determine if we need to threshold
        i.e. if num_peaks is None, then no thresholding
    total_peaks : int
        Total number of peaks in this chromosome
    to_remove : list
        List of chromosome names to remove
    chrom_name : str
        Name of the chromosome being processed
    base_ratio : float
        The ratio of peaks to keep based on base chromosome

    Returns
    -------
    p_array_thresh : np.ndarray
        Thresholded peak array
    to_remove : list
        Updated list of chromosome names to remove
    """
    if num_peaks is not None:
        # if num_peaks is specified, filter the peaks otherwise keep all peaks
        num_keep = int(total_peaks * base_ratio)
        if num_keep == 0:
            log.warning(f'Removing {chrom_name} since the number to keep based on base ratio {base_ratio:.3g}/{total_peaks} is zero')
            to_remove.append(chrom_name)
            threshold = np.inf
        elif num_keep == total_peaks:
            log.info(f'Keeping all {total_peaks} peaks in {chrom_name}')
            threshold = -np.inf
        else:
            log.info(f'Keeping {num_keep}/{total_peaks} peaks in {chrom_name} based on base ratio {base_ratio:.3g}')
            threshold = np.partition(p_array, -num_keep)[-num_keep]
    
        # Apply thresholding here
        p_array_thresh = p_array.copy() # Avoid modifying original
        p_array_thresh[p_array_thresh < threshold] = 0.0
    
    return p_array_thresh, to_remove

def read_bedpe_and_bin(bedpe_file, bin_size, min_hic_value):
    """
    Read in Hi-C bedpe file and bin the interactions according to bin_size

    Code is almost identical to the legacy code in GenomeLoopData with a few extra checks

    Returns
    -------
    bedpe_data : dict
        Key: chrom name
        Value: np.array of shape (num_interactions, 3) where each row is (b1, b2, val)
        b1 and b2 are the binned positions of the two anchors
        val is the interaction value
    hic_chromosomes : list
        List of chromosomes found in the bedpe file
    """
    bedpe_data = {}
    with open(bedpe_file) as in_file:
        for line in in_file:
            if line.startswith("#"):
                continue # Header line

            fields = line.strip().split()

            if len(fields) < 7:
                continue

            chrom1 = fields[0]
            chrom2 = fields[3]

            if chrom1 != chrom2:
                continue # Ignore inter-chromosomal

            val = float(fields[6])

            if val < min_hic_value:
                continue
            
            # Head anchor
            s1 = int(fields[1])
            e1 = int(fields[2])
            # Tail anchor
            s2 = int(fields[4])
            e2 = int(fields[5])

            # OK to floor because bedpe is 0-based start, 1-based end
            mp1 = (s1 + e1) // 2
            mp2 = (s2 + e2) // 2

            # Bin the midpoints of the anchors
            b1 = mp1 // bin_size
            b2 = mp2 // bin_size

            if chrom1 not in bedpe_data:
                bedpe_data[chrom1] = []

            bedpe_data[chrom1].append((b1, b2, val))
            
        hic_chromosomes = list(bedpe_data.keys())
        for k in bedpe_data:
            bedpe_data[k] = np.array(bedpe_data[k])

        return bedpe_data, hic_chromosomes
    

class GenomeBinData:
    """
    A class used to represent a sample

    Attributes
    ----------
    species_name : str
        Name of the sample's species (hg38, mm10, ...)
    sample_name : str
        Name of the sample (LHH0061, LHH0061_0061H, ...)
    chrom_dict : dict[str, ChromBinData]
        Key: Name of chromosome
        Value: ChromBinData object
    peak_dict : dict[str, np.ndarray]
        Key: Name of chromosome
        Value: 1D numpy array of number of bins in chromosome. 
        Each entry represents the peak value for that bin
    """

    def __init__(
        self,
        chrom_size_file: str,
        bedgraph_file: str,
        chrom_structure_file: str,
        window_size: int,
        window_stride: int,
        normalization: str = 'NONE',
        peak_dict: Dict[str, list] = None,
        from_peak_file: bool = None,
        base_chrom: str = 'chr1',
        num_peaks: int = None,
        chroms_to_load: List[str] = None,
        bin_size: int = 1,
        min_hic_value: int = 1,
        min_bedgraph_value: int = 1,
        ba_mult: int = 1
    ):
        """
        Loads in chromatin structure, binding affinity, and potentially peaks

        Data is standardized to all be binned at `bin_size` resolution

        Binding affinity is used to weigh the adjacency matrices, which may result in values
        less than 1 if binding affinity values are less than 1.

        This will affect method="diffusion" with laplacian computation since values less than 1 are filtered out

        Removes chromosomes that were problematic during loading

        Parameters
        ----------
        chrom_size_file : str
            File containing the base pair size of each chromosome to use
        bedgraph_file : str
            File containing binding affinity for this sample
        chrom_structure_file : str
            File containing chromatin structure. 
            This can be a .hic file or a .bedpe file:
            chrom1  start1   end1 chrom2  start2   end2 pet_count
        window_size : int
            Size of sliding window
        window_stride : int
            Factor of window_size to stride (e.g. 2 means stride by window_size/2)
        normalization : str, optional
            Hi-C normalization method. 
            Options are same as .hic file normalization methods (e.g. NONE, VC, VC_SQRT, KR)
            WARNING: If the normalization results in values that are less than 1,
            then those values will be filtered out by two mechanisms:

            (1) min_hic_value
            (2) chrom_bin_data.py: `laplacian` function which considers values ≥ 1

        peak_dict : dict, optional
            Key: chrom name
            Value: list of (start, end, length) tuples for each peak or empty list
            If None, then no peaks are used (no peak filtering)
        from_peak_file : bool, optional
            Whether the peak_dict is constructed from a peak file (True) or
            to be constructed from bedgraph node weights directly (False).
            Only used if peak_dict is not None.
        base_chrom : str
            Chromosome to use when selecting number of peaks        
        num_peaks : int, optional
            Number of peaks to keep in `base_chrom`
            The same ratio of peaks will be kept for other chromosomes
            If None, keep all peaks
        chroms_to_load : list, optional
             List of names of chromosome to load (default is None)
        bin_size : int
            Binning resolution 
        min_hic_value : int
            Minimum Hi-C value to consider a valid interaction
        min_bedgraph_value : int
            Minimum bedgraph value to consider a valid binding affinity
        ba_mult : int
            Multiplicative factor to multiply binding affinity values by
            This is recommended if there are binding affinity values are less than 1 
            (i.e. non-integer values between 0 and 1)
        """
        self.species_name = os.path.basename(chrom_size_file).split('.')[0]
        self.sample_name = os.path.basename(chrom_structure_file).split('.')[0]
        # Iterate through all chromosomes once because we need all data anyways
        # for this, we read in the chrom size file first

        # Peak objects
        self.peak_dict = {}
        base_ratio = None

        # Initialize all chromosomes to be loaded
        self.chrom_dict = {}
        with open(chrom_size_file) as in_file:
            for line in in_file:
                line = line.strip().split()
                chrom_name = line[0]
                if chroms_to_load and chrom_name not in chroms_to_load:
                    continue

                if chrom_name in CHROMS_TO_IGNORE:
                    continue

                chrom_size = int(line[1])

                self.chrom_dict[chrom_name] = \
                    ChromBinData(chrom_name, chrom_size, self.sample_name, bin_size, window_size, window_stride)
                
        # Chromosomes to remove (if either bedgraph or hic data is missing)
        to_remove = []

        # Read in binding affinity data
        bedgraph = BedGraph(chrom_size_file, bedgraph_file, 
                            chroms_to_load=chroms_to_load, 
                            ignore_missing_bp=False, 
                            min_value=min_bedgraph_value)
        
        # Read in chromatin structure data
        is_bedpe = chrom_structure_file.endswith('.bedpe')
        if is_bedpe:
            # Bedpe mode
            bedpe_data, hic_chromosomes = read_bedpe_and_bin(chrom_structure_file, bin_size, min_hic_value)
        else: 
            # Hi-C mode
            hic = hicstraw.HiCFile(chrom_structure_file)
            if bin_size not in hic.getResolutions():
                log.error(f'Bin size {bin_size} not found in {chrom_structure_file} resolutions: {hic.getResolutions()}')
                raise ValueError(f'Bin size {bin_size} not found in {chrom_structure_file} resolutions: {hic.getResolutions()}')
            hic_chromosomes = [x.name for x in hic.getChromosomes()]
            log.info(f'Chromosomes detected in {chrom_structure_file}: {hic_chromosomes}')

        # Move base_chrom to the front for processing first
        # This is important for peak threshold computation, where we need base_chrom's ratio of peaks first
        if base_chrom in self.chrom_dict:
            self.chrom_dict = {
                base_chrom: self.chrom_dict[base_chrom],
                **{k: v for k, v in self.chrom_dict.items() if k != base_chrom}
            }
        else:
            log.warning(f'Base chromosome {base_chrom} not found in chrom_dict')
            base_chrom = list(self.chrom_dict.keys())[0] # First chrom available
            log.warning(f'Using {base_chrom} as base chromosome instead')
            # No need to reorder since base_chrom is the first one already

        # Common for loop
        for chrom_name, chrom_data in self.chrom_dict.items():

            # BINDING AFFINITY
            if not bedgraph.has_chrom(chrom_name):
                to_remove.append(chrom_name)
                continue
            
            # Load chrom data once (for peaks and node features)
            bedgraph.load_chrom_data(chrom_name)

            if chroms_to_load and chrom_name not in chroms_to_load:
                continue

            # Compute binned binding affinity as the node features for the whole chromosome
            start_list = np.arange(0, chrom_data.size, bin_size, dtype=np.int32)
            end_list = start_list + bin_size

            # Last bin might be smaller than chrom size
            np.clip(end_list, 0, chrom_data.size, out=end_list)

            node_weights = bedgraph.stats(start_list=start_list, end_list=end_list, 
                                           stat="max", chrom_name=chrom_name)
            np.nan_to_num(node_weights, nan=0.0, copy=False)

            if ba_mult != 1:
                node_weights *= ba_mult

            window_start_idx = chrom_data.window_indices[:, 0]
            window_end_idx = chrom_data.window_indices[:, 1]

            b = []
            for i in range(len(window_start_idx)):
                b.append(node_weights[window_start_idx[i]:window_end_idx[i]])
            b = np.array(b, dtype=np.float32) # MxN array
            log.info(f'Chromosome {chrom_name} node weights shape (MxN): {b.shape}')
            chrom_data.node_weights = b

            # PEAKS
            # Find values for each peak since each peak caller is not accurate sometimes
            # We should only do this if peak_dict is passed in from a peak file
            # Otherwise, we should construct from node weights directly
            if peak_dict is not None:

                peak_chromosomes = list(peak_dict.keys())
                if chrom_name not in peak_chromosomes:
                    continue

                # Initialize peak array
                p = np.zeros_like(node_weights)

                if from_peak_file:
                # Interpret as a valid peak file and resort to the original code
                # where we extract the max value from the bedgraph for each peak
                    peak_chrom = peak_dict[chrom_name]

                    start_list = [x[0] for x in peak_chrom]
                    end_list = [x[1] for x in peak_chrom]
                    max_list = bedgraph.stats(start_list=start_list, end_list=end_list,
                                              chrom_name=chrom_name, stat='max')
                    total_peaks = len(max_list)
                    
                    if chrom_name == base_chrom: # Guaranted to enter in first iteration
                        base_ratio = num_peaks / total_peaks if total_peaks > 0 else 1.0
                        base_ratio = np.clip(base_ratio, 0.0, 1.0)
                    
                    # Threshold peaks here by zeroing out the values of the peaks (i.e. `max_list`)
                    max_list, to_remove = threshold_peaks(np.array(max_list), num_peaks=num_peaks,
                                                          total_peaks=total_peaks, to_remove=to_remove,
                                                          chrom_name=chrom_name, base_ratio=base_ratio)

                    # Then construct the peak array using thresholded max_list
                    for i in range(len(start_list)):
                        s_idx = int(start_list[i] // bin_size) # inclusive start
                        e_idx = int((end_list[i] - 1) // bin_size) + 1 # exclusive end
                        
                        # Bound checks
                        s_idx = max(0, s_idx)
                        e_idx = min(len(p), e_idx)

                        if e_idx > s_idx:
                            # Only assign if peak has non-zero length
                            p[s_idx:e_idx] += max_list[i]

                else:
                    # Use `node_weights` directly as peaks
                    p = node_weights.copy()
                    total_peaks = len(p)

                    if chrom_name == base_chrom: # Guaranted to enter in first iteration
                        base_ratio = num_peaks / total_peaks if total_peaks > 0 else 1.0
                        base_ratio = np.clip(base_ratio, 0.0, 1.0)
                    
                    # Threshold peaks here
                    p, to_remove = threshold_peaks(p, num_peaks=num_peaks, total_peaks=total_peaks,
                                                   to_remove=to_remove, chrom_name=chrom_name,
                                                   base_ratio=base_ratio)

                self.peak_dict[chrom_name] = p

            # Free data for chromosome
            bedgraph.free_chrom_data(chrom_name)


            # CHROMATIN STRUCTURE
            chrom_name_hic = check_alt_chrom_name(chrom_name, hic_chromosomes)
            # First check if the current chrom_name exists in hic_chromosomes
            if chrom_name_hic is None:
                to_remove.append(chrom_name)
                continue

            window_start = chrom_data.window_locations[:, 0]
            window_end = chrom_data.window_locations[:, 1]
            A = []

            if is_bedpe:
                loops = bedpe_data[chrom_name_hic]

                for i in range(len(window_start)):
                    window_start_bin = window_start[i] // bin_size
                    window_end_bin = window_end[i] // bin_size
                    N = window_end_bin - window_start_bin

                    mat = np.zeros((N, N), dtype=np.float32)

                    if loops.shape[0] > 0:
                        # Select the loops that fall within current window
                        mask = (loops[:, 0] >= window_start_bin) & (loops[:, 0] < window_end_bin) & \
                               (loops[:, 1] >= window_start_bin) & (loops[:, 1] < window_end_bin)
                        
                        window_loops = loops[mask]

                        for row in window_loops:
                            b1 = int(row[0])
                            b2 = int(row[1])
                            val = row[2]

                            r = b1 - window_start_bin
                            c = b2 - window_start_bin

                            mat[r, c] += val
                            if r != c:
                                mat[c, r] += val # Symmetric
                    
                        nnz = np.count_nonzero(mat)
                        if nnz == 0:
                            log.warning(f'No interactions found in {chrom_name} for window {i} ({window_start[i]}-{window_end[i]})')
                            
                        A.append(mat)

            else:
                # Hi-C mode
                mzd = hic.getMatrixZoomData(chrom_name_hic, chrom_name_hic, "observed", normalization, "BP", int(bin_size))

                for i in range(len(window_start)):
                    mat = mzd.getRecordsAsMatrix(
                        window_start[i] - bin_size, window_end[i] - bin_size, # Extract one pixel beyond the start always first row and col is always 0
                        window_start[i] - bin_size, window_end[i] - bin_size # Because end is inclusive
                        )
                    
                    if mat.shape[1] == 1:
                        # Assume that any 1x1 matrix is empty
                        log.warning(f'No Hi-C data found in {chrom_name} for window {i} ({window_start[i]}-{window_end[i]})')
                        N = min(np.floor(chrom_data.size / bin_size).astype(int), np.floor(window_size / bin_size).astype(int))
                        mat = np.zeros((N, N))
                    else:
                        mat = mat[1:, 1:] # Remove first row and column which are always 0

                        # Filter by min_hic_value
                        nnz_before = np.count_nonzero(mat)
                        mat[mat < min_hic_value] = 0
                        nnz_after = np.count_nonzero(mat)

                        if nnz_before != nnz_after:
                            log.info(f'Removing interactions with values below {min_hic_value}: {nnz_before} -> {nnz_after} out of {mat.shape[0]*mat.shape[1]}')
                        if nnz_after == 0:
                            log.warning(f'All entries in {chrom_name} are below {min_hic_value}')

                        if window_end[i] - window_start[i] != chrom_data.window_locations[0, 1] - chrom_data.window_locations[0, 0]:
                            log.error('Error in constructing adjacency matrices: Window size mismatch')
                            raise ValueError('Error in constructing adjacency matrices: Window size mismatch')

                    A.append(mat)

            A = np.array(A, dtype=np.float32) # MxNxN array
            log.info(f'Chromosome {chrom_name} adjacency matrices shape (MxNxN): {A.shape}')

            # WEIGH BY BINDING AFFINITY
            A = weigh_adjacency_matrix_binding_affinity(A, b)

            # List of pointers to sparse matrices
            chrom_data.adjacency_matrices = [coo_matrix(A[i]) for i in range(A.shape[0])]
        
        # Get rid of chroms that had problems initializing
        # Chromosomes with no loops or other random problems
        for chrom_name in to_remove:
            if chrom_name in self.chrom_dict:
                del self.chrom_dict[chrom_name]

        for chrom_name in self.chrom_dict:
            # Do final size consistency check between node features and adjacency matrices
            self.chrom_dict[chrom_name].finish_init()


    def preprocess(
        self,
        both_peak_support: bool = False,
        output_dir: str = 'output',
        diagnostic_plots: bool = True,
    ) -> None:
        """
        Filters out Hi-C contacts without peak support using matrix operations

        Removes chromosomes that result in all zero matrices after filtering

        Optionally saves diagnostic plots of peak array

        Parameters
        ----------
        both_peak_support : bool, optional
            Whether to only keep loops that have peak support on both sides
        output_dir : str, optional
            Directory to output found peaks and filters
        diagnostic_plots : bool, optional
            Whether to save diagnostic plots of peak array
        Returns
        ------
        None
        """

        os.makedirs(f'{output_dir}/peaks', exist_ok=True)

        bp_eng = EngFormatter(unit='bp', places=0, sep="\N{THIN SPACE}")

        def genomic_tick_formatter(window_start, bin_size):
            # Returns a FuncFormatter that maps index -> genomic coordinate
            return FuncFormatter(lambda x, pos: bp_eng(window_start + x * bin_size))
        
        to_remove = []
        for chrom_name, chrom_data in self.chrom_dict.items():

            if not self.peak_dict or chrom_name not in self.peak_dict:
                # No peaks to preprocess
                continue

            success = chrom_data.filter_with_peaks(self.peak_dict[chrom_name],
                                                   both_peak_support=both_peak_support)
            
            if not success:
                to_remove.append(chrom_name)
            elif diagnostic_plots:
                peak_array = self.peak_dict[chrom_name]

                fig, ax = plt.subplots(layout="constrained", figsize=(12, 3))
                ax.stem(np.arange(len(peak_array)), peak_array, markerfmt='.', basefmt=" ")
                
                ax.xaxis.set_major_formatter(genomic_tick_formatter(0, self.chrom_dict[chrom_name].bin_size))
                ax.set_title(f"{self.sample_name} {chrom_name} Peaks")
                
                plt.savefig(f'{output_dir}/peaks/{self.sample_name}_{chrom_name}.png', dpi=150)
                plt.close(fig)

        # Remove problematic chromosomes
        for chrom_name in to_remove:
            del self.chrom_dict[chrom_name]





    def compare(
            self, 
            o_loop_data: 'GenomeBinData',
            alpha: float = 0.5,
            method: str = 'fgw',
            cost : str = 'linear',
            chroms_to_compare: List[str] = None,
            output_dir: str = 'output',
            do_output_graph: bool = False,
            gvmax: float = 100.0,
            scale_cost: str = 'unitless',
            mass: float = 1.0,
            feat: str = 'index_BA',
            weight: str = 'uniform',
            mu: Union[float, int] = 1.0,
            compare_method: str = 'spearman',
            cross: bool = False,
            ssp: float = None,
            num_cores: int = 1,
            param_str: str = '',
    ) -> Dict[str, float]:
        """
        Compares this sample (genome bin data) to another sample (genome bin data)

        Gets the comparison values for each window in each chromosome and
        combines that into a genome-wide comparison value. Each window is given
        a weight based on the highest loop in it. Each chromosome is weighted
        equally
        """
        # Default: Compare all the chromosomes
        if chroms_to_compare is None:
            chroms_to_compare = list(self.chrom_dict.keys())

        comparison_name = f'{self.sample_name}_{o_loop_data.sample_name}'
        os.makedirs(f'{output_dir}/{param_str}/comparisons/{comparison_name}',
                    exist_ok=True)
        os.makedirs(f'{output_dir}/{param_str}/scores/windows', exist_ok=True)
        os.makedirs(f'{output_dir}/{param_str}/scores/chromosomes',
                    exist_ok=True)
    
        chrom_score_dict = {}
        log.info(f'Chromosomes to compare: {chroms_to_compare}')
        for chrom_name in chroms_to_compare:

            if chrom_name not in self.chrom_dict:
                log.warning(f'{chrom_name} is not in {self.sample_name}. '
                            f'Skipping {chrom_name}')
                continue
        
            if chrom_name not in o_loop_data.chrom_dict:
                log.warning(f'{chrom_name} is in {self.sample_name} but '
                            f'not in {o_loop_data.sample_name}. Skipping '
                            f'{chrom_name}')
                continue

            log.info(f"Comparing {chrom_name} ...")

            # Returns a list of distance values for each window comp
            dist_values_windows, weights, max_graph, o_max_graph = self.chrom_dict[chrom_name].compare(
                o_chrom=o_loop_data.chrom_dict[chrom_name],
                alpha=alpha,
                method=method,
                cost=cost,
                output_dir=output_dir,
                do_output_graph=do_output_graph,
                param_str=param_str,
                gvmax=gvmax,
                scale_cost=scale_cost,
                mass=mass,
                feat=feat,
                weight=weight,
                mu=mu,
                compare_method=compare_method,
                cross=cross,
                ssp=ssp,
                num_cores=num_cores
            )
            chrom_comp_values = []
            # Index 0: Weighted average
            try:
                chrom_comp_values.append(nan_average(dist_values_windows, weights=weights))
            except ZeroDivisionError:
                log.exception(f"No loops were found in either graphs."
                              "FGW should resort to just OT of node features"
                              f"{chrom_name}")
                chrom_comp_values.append(np.nan)

            # Index 1: Unweighted average
            chrom_comp_values.append(nan_average(dist_values_windows))

            chrom_score_dict[chrom_name] = chrom_comp_values
            log.debug(f'{chrom_name} comp values: {chrom_comp_values}')

            window_starts = self.chrom_dict[chrom_name].window_locations[:, 0]
            window_ends = self.chrom_dict[chrom_name].window_locations[:, 1]

            with open(f'{output_dir}/{param_str}/scores/windows/'
                      f'{comparison_name}_{chrom_name}.txt', 'w') as out_file:
                out_file.write(
                    f'chrom_name\twindow_start\twindow_end\t'
                    f'{method}\twindow_weights\tmax_graph\to_max_graph\n')
                for i in range(len(dist_values_windows)):
                    window_start = window_starts[i]
                    window_end = window_ends[i]
                    dist_value = dist_values_windows[i]
                    weight = weights[i]
                    max_window = max_graph[i]
                    o_max_window = o_max_graph[i]
                    out_file.write(f'{chrom_name}\t{window_start}\t{window_end}'
                                   f'\t{dist_value}\t{weight}\t{max_window}\t{o_max_window}\n')

        with open(f'{output_dir}/{param_str}/scores/chromosomes/'
                  f'{comparison_name}.txt', 'w') as out_file:
            out_file.write(f'chrom_name\t'
                        f'{method}_weighted\t{method}_unweighted\n')
            for chrom_name, score_dict in chrom_score_dict.items():
                dist_value_weighted = score_dict[0]
                dist_value_unweighted = score_dict[1]
                out_file.write(f'{chrom_name}\t'
                            f'{dist_value_weighted}\t{dist_value_unweighted}\n')

        # Average across each chromosome equally    
        avg_value = {
            "dist_weighted" : np.mean([x[0] for x in chrom_score_dict.values()]),
            "dist_unweighted" : np.mean([x[1] for x in chrom_score_dict.values()]),
        }
        log.debug(avg_value)

        return avg_value

