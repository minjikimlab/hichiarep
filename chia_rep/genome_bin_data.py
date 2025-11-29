import numpy as np
import os
import logging
import math
from pyBedGraph import BedGraph
import hicstraw
from typing import Dict, List, Union
from scipy.sparse import coo_matrix

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
    Checks for alternate chromosome names (chr1 vs 1)

    Parameters
    ----------
    chrom_name : str
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
    A[i,j] = A[i,j] * (b[i] + b[j])
    """
    out = A * b[:, :, None]
    out += A * b[:, None, :]
    return out


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
    """

    def __init__(
        self,
        chrom_size_file: str,
        bedgraph_file: str,
        hic_file: str,
        window_size: int,
        window_stride: int,
        normalization: str = 'NONE',
        chroms_to_load: List[str] = None,
        bin_size: int = 1,
        min_hic_value: int = 1,
        min_bedgraph_value: int = 1
    ):
        """
        Initializes all chromosomes and adds loops to them from given file.

        Finds peak max from bedgraph

        Parameters
        ----------
        chrom_size_file : str
            File containing the base pair size of each chromosome to use
        loop_file : str
            File containing loops in format:
            chrom1  start1   end1 chrom2  start2   end2 pet_count
        bedgraph : BedGraph
            The bedgraph file for this sample (from pyBedGraph)
        chroms_to_load : list, optional
             List of names of chromosome to load (default is None)
        min_loop_value : int, optional
            Minimum loop value (PET count) to include (default is 0)
        bin_size : int
            Bin size of bedgraph for FGW
        """
        self.species_name = os.path.basename(chrom_size_file).split('.')[0]
        self.sample_name = os.path.basename(hic_file).split('.')[0]
        # Iterate through all chromosomes once because we need all data anyways
        # for this, we read in the chrom size file first

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
        hic = hicstraw.HiCFile(hic_file)
        if bin_size not in hic.getResolutions():
            log.error(f'Bin size {bin_size} not found in {hic_file} resolutions: {hic.getResolutions()}')
            raise ValueError(f'Bin size {bin_size} not found in {hic_file} resolutions: {hic.getResolutions()}')
        hic_chromosomes = [x.name for x in hic.getChromosomes()]
        log.info(f'Chromosomes detected in {hic_file}: {hic_chromosomes}')

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
            num_bins = np.ceil(chrom_data.size / bin_size).astype(int)
            start_list = np.arange(num_bins, dtype=np.int32) * bin_size
            end_list = (np.arange(num_bins, dtype=np.int32) + 1) * bin_size

            # Last bin might be smaller than chrom size
            np.clip(end_list, 0, chrom_data.size, out=end_list)

            node_weights = bedgraph.stats(start_list=start_list, end_list=end_list, 
                                           stat="max", chrom_name=chrom_name)
            np.nan_to_num(node_weights, nan=0.0, copy=False)

            window_start_idx = chrom_data.window_indices[:, 0]
            window_end_idx = chrom_data.window_indices[:, 1]

            b = []
            for i in range(len(window_start_idx)):
                b.append(node_weights[window_start_idx[i]:window_end_idx[i]])
            b = np.array(b, dtype=np.float32) # MxN array
            log.info(f'Chromosome {chrom_name} node weights shape (MxN): {b.shape}')
            chrom_data.node_weights = b

            # Free data for chromosome
            bedgraph.free_chrom_data(chrom_name)



            # CHROMATIN STRUCTURE
            chrom_name_hic = check_alt_chrom_name(chrom_name, hic_chromosomes)

            if chrom_name_hic is None:
                to_remove.append(chrom_name)
                continue

            mzd = hic.getMatrixZoomData(chrom_name_hic, chrom_name_hic, "observed", normalization, "BP", int(bin_size))

            window_start = chrom_data.window_locations[:, 0]
            window_end = chrom_data.window_locations[:, 1]

            A = []
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

    def compare(
            self, 
            o_loop_data: 'GenomeBinData',
            window_size: int,
            window_stride: int,
            bin_size: int,
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
            num_cores: int = 1
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
        param_str = f'{window_size}.{bin_size}.{method}'
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
                o_loop_data.chrom_dict[chrom_name],
                alpha,
                method,
                cost,
                output_dir,
                do_output_graph,
                param_str,
                gvmax=gvmax,
                scale_cost=scale_cost,
                mass=mass,
                feat=feat,
                weight=weight,
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

