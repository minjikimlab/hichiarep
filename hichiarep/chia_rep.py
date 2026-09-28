import os
import re
import math
from collections import OrderedDict
import time
import csv
import logging
from typing import Dict, List, Tuple, Union
# import pybedtools

# from .genome_loop_data import GenomeLoopData
from .genome_bin_data import GenomeBinData, WINDOW_WEIGHTINGS, check_alt_chrom_name, read_chrom_sizes, \
    standardize_chrom_name
import numpy as np

log = logging.getLogger()
# score_dict = OrderedDict[str, OrderedDict[str, float]]
score_dict = Dict[str, Dict[str, float]]

# Limits of the single window compared for a genomic_location (see `parse_genomic_location`)
MIN_REGION_SPAN = 20000 # 20 kb, minimum span of the requested region
MIN_REGION_BINS = 10     # The window matrix is at least 10 x 10
MAX_REGION_BINS = 1000  # and at most 1000 x 1000

# 'chrom:start-end', commas allowed in the coordinates
GENOMIC_LOCATION_RE = re.compile(r'([^:\s]+):(\d[\d,]*)-(\d[\d,]*)')


def output_score(
    scores: score_dict,
    file_path: str
) -> None:
    """
    Output scores to a specified file path.

    Parameters
    ----------
    scores
    file_path

    Returns
    -------
    None
    """
    with open(f'{file_path}', 'w') as out_file:
        header = ['Sample Name'] + list(scores.keys())
        writer = csv.DictWriter(out_file, fieldnames=header)
        writer.writeheader()
        for sample_name, sample_scores in scores.items():
            writer.writerow(sample_scores)


def output_to_csv(
    weighted_scores: Dict[str, score_dict],
    unweighted_scores: score_dict,
    output_dir: str = 'output',
    param_str: str = None
) -> None:
    """
    Outputs weighted and unweighted scores to a specified file path.

    Parameters
    ----------
    weighted_scores
        Key: window weighting (currently only 'beta'), value: scores.
        Each is written to {weighting}_weighted_scores.csv
    unweighted_scores
    output_dir
        Directory to output scores

    Returns
    ------
    None
    """
    score_dir = f'{output_dir}/{param_str}/scores'
    os.makedirs(score_dir, exist_ok=True)

    if weighted_scores is not None:
        # Weighted scores, one file per window weighting
        for weighting, scores in weighted_scores.items():
            output_score(scores, f'{score_dir}/{weighting}_weighted_scores.csv')

    if unweighted_scores is not None:
        # Unweighted scores
        output_score(unweighted_scores, f'{score_dir}/unweighted_scores.csv')

    log.info(f"Results have been written to {score_dir}")


def compare(
    sample_dict: OrderedDict,
    method: str = 'random_walk',
    cost: str = 'linear',
    compare_list: list = None,
    compare_list_file: str = None,
    output_dir: str = 'output',
    alpha: float = 0.5,
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
    param_str: str = None,
    report_conv_it: bool = False,
    live_bin_rule: str = 'either',
    min_live_bins: int = 10,
    shapley: bool = False
) -> (score_dict, Dict[str, score_dict]):
    """
    Compares specified samples against each other. Specify comparisons in either
    compare_list or compare_list_file.

    Parameters
    ----------
    sample_dict : OrderedDict
        (Key: sample name, value: sample data)
    method : str
        Comparison method to use. Options are:

        - 'random_walk'
        - 'diffusion'

        Deprecated options: 

        - 'w' (wasserstein)
            Associated parameters: `feat`, `weight`
        - 'fgw' (fused gromov-wasserstein)
            Associated parameters: `alpha`, `cost`, `gvmax`, `scale_cost`, `mass`
        - 'JSD' (legacy JSD)
        - 'EMD' (legacy EMD)
    compare_list : list, optional
        List of comparisons to make. Shape is n x 2 where n is the number of
        comparisons
    compare_list_file : str, optional
        File that contains a list of comparisons to make.
        Format:
        sample1_name    sample2_name
        sample3_name    sample4_name
        ...
    output_dir : str
        Directory to output data
    do_output_graph : bool
        Output processed .npy graphs for each window of chromosome 1 only
        This is mainly for developer debugging purposes
    mu : float or int
        Power to raise the random walk transition matrix (if method is 'random_walk')
        Diffusion time step (if method is 'diffusion')
    compare_method : str
        Method to compare processed binding affinity signals 
        after passing through random walk or diffusion.
        Either 'spearman' or 'jsd'
    cross : bool
        Whether to compare the signals 'direct' or 'cross'.

        Let (x1, K1) be the binding affinity and linear operator for sample 1
        Let (x2, K2) be the binding affinity and linear operator for sample 2

        Then,

        Direct comparison:
        x1 -> K1 -> x11
        x2 -> K2 -> x22
        Compare x11 and x22

        Cross comparison:
        x1 -> K2 -> x12
        x2 -> K1 -> x21
        Compare x11 and x12
        Compare x21 and x22
        Take average of the two comparisons
    ssp : float
        Subsample percentage in [0, 1]. If None, then no subsampling is done.
        If specified as a float between 0 and 1, then that initial subsampling is done
        to ensure the read depth is identical between any two pairs.
        Additional subsampling is done that is a proportion of this common read depth.
        The subsampling of every window is seeded by the pair, the chromosome and the window,
        so it is reproducible and does not depend on `num_cores`
    num_cores : int
        Number of pools to use for parallel processing across the windows of a chromosome.
        Ignored for samples loaded for a genomic_location, which hold a single window
    param_str : str
        Parameter string that is the subfolder name under output_dir
    report_conv_it : bool
        Whether to report the iteration at which the random walk of each sample
        converged, i.e. the smallest k with

            ||P^(k+1) - P^k||_F / ||P^k||_F < 1e-6

        Written as two extra columns in the per-window score files.
        Only applies to method='random_walk'
    live_bin_rule : str
        A bin is dead on one side of a window pair if its column of the coverage
        weighted adjacency matrix (before the blur) sums to 0. Dead bins are removed
        (rows / columns of A and entries of the binding affinity, on both sides)
        before the random walk:

        - 'either': drop bins dead on either side (keep the intersection of the live bins)
        - 'both': drop bins dead on both sides only (keep the union of the live bins)
        - 'none': keep every bin

        Only applies to method='random_walk'
    min_live_bins : int
        Windows with fewer live bins are skipped (nan).
        Only applies to method='random_walk'
    shapley : bool
        Whether to also compute the Shapley attribution of every window score W_SB.
        Each window pair is scored twice more on the same live bins: W_S with the binding
        affinity replaced by ones (structure only) and W_B with the chromatin structure
        replaced by the identity (binding only). The per-window score files gain the
        columns W_S, W_B and alpha_shapley, the share of W_SB attributed to structure
        (nan if |W_SB| <= 0.1 or if a Shapley value is negative). Roughly triples the time
        of the comparisons, and the samples must be loaded with read_data(..., shapley=True).
        Only applies to method='random_walk' without subsampling (ssp=None)
    Returns
    -------
    OrderedDict containing unweighted scores, and a dict (key: window weighting) of
    OrderedDicts containing weighted scores. The only window weighting is

    - 'beta': beta_i = n_live / N, the fraction of bins kept by the live bin mask
      (all ones for methods other than 'random_walk')

    normalized to a PMF over the scored windows of a chromosome
    """
    if shapley and (method != 'random_walk' or ssp is not None):
        log.error(f'The Shapley attribution is only implemented for method="random_walk" without '
                  f'subsampling, got method="{method}" and ssp={ssp}')
        raise ValueError(f'The Shapley attribution is only implemented for method="random_walk" without '
                         f'subsampling, got method="{method}" and ssp={ssp}')

    # Samples loaded for a genomic_location hold a single window, so there are no windows to
    # parallelize across. Scored in the main process, the window also keeps every BLAS thread
    # for its dense random walk instead of the one thread of a pool worker
    if num_cores > 1 and any(sample.region is not None for sample in sample_dict.values()):
        log.info(f'Single window (genomic_location): ignoring num_cores={num_cores}')
        num_cores = 1

    total_start_time = time.time()
    os.makedirs(f'{output_dir}/timings', exist_ok=True)
    sample_list = list(sample_dict.keys())

    if compare_list is None:
        if compare_list_file is None or not os.path.isfile(compare_list_file):
            log.error(f"{compare_list_file} is not a valid file")
            return OrderedDict(), OrderedDict()

        to_compare_list = []
        with open(compare_list_file) as in_file:
            for line in in_file:
                comparison = line.split()
                if len(comparison) != 2:
                    log.error(f'Invalid number of columns in {compare_list_file}')
                    return OrderedDict(), OrderedDict()

                to_compare_list.append(comparison)
    else:
        for comparison in compare_list:
            if len(comparison) != 2:
                log.error(f'Invalid list length in {comparison}')
                return OrderedDict(), OrderedDict()

        to_compare_list = compare_list

    # To easily output in .csv format
    # One score table per window weighting (beta)
    scores_weighted = OrderedDict((weighting, OrderedDict()) for weighting in WINDOW_WEIGHTINGS)
    scores_unweighted = OrderedDict()

    for key in sample_list:
        # The new distances are distances not similarities
        # So diagonal is 0
        for scores in scores_weighted.values():
            scores[key] = OrderedDict()
            scores[key][key] = 0
            scores[key]['Sample Name'] = key

        scores_unweighted[key] = OrderedDict()
        scores_unweighted[key][key] = 0
        scores_unweighted[key]['Sample Name'] = key


    comparison_timings = OrderedDict()
    for comparison in to_compare_list:
        comparison_start_time = time.time()

        sample1_name, sample2_name = comparison
        sample1 = sample_dict[sample1_name]
        sample2 = sample_dict[sample2_name]
        comparison_name = f'{sample1_name}_{sample2_name}'
        log.info(f'Compare {comparison_name}')

        if sample1.species_name != sample2.species_name:
            log.error('Tried to compare two different species. Skipping')

        value_dict = sample1.compare(sample2,
                                     alpha=alpha, 
                                     method=method, 
                                     mu=mu,
                                     compare_method=compare_method,
                                     cross=cross,
                                     output_dir=output_dir,
                                     do_output_graph=do_output_graph,
                                     cost=cost,
                                     gvmax=gvmax,
                                     scale_cost=scale_cost,
                                     mass=mass,
                                     feat=feat,
                                     weight=weight,
                                     ssp=ssp,
                                     num_cores=num_cores,
                                     param_str=param_str,
                                     report_conv_it=report_conv_it,
                                     live_bin_rule=live_bin_rule,
                                     min_live_bins=min_live_bins,
                                     shapley=shapley)

        # Save values in OrderedDict
        for weighting, scores in scores_weighted.items():
            dist_weighted = value_dict[f'dist_{weighting}_weighted']
            scores[sample1_name][sample2_name] = dist_weighted
            scores[sample2_name][sample1_name] = dist_weighted

        dist_unweighted = value_dict['dist_unweighted']

        scores_unweighted[sample1_name][sample2_name] = dist_unweighted
        scores_unweighted[sample2_name][sample1_name] = dist_unweighted

        weighted_str = ', '.join(f'{w}: {value_dict[f"dist_{w}_weighted"]:.3g}' for w in WINDOW_WEIGHTINGS)
        log.info(f'{comparison_name} {method}: {dist_unweighted:.3g} (weighted {weighted_str})')

        comparison_timings[comparison_name] = time.time() - comparison_start_time

    with open(f'{output_dir}/timings/comparison.{param_str}.txt',
              'w') as out_file:
        out_file.write(f'comparison\ttime_taken\n')
        for comparison_name, compare_timing in comparison_timings.items():
            out_file.write(f'{comparison_name}\t{compare_timing}\n')
        out_file.write(f'total\t{time.time() - total_start_time}\n')

    return scores_unweighted, scores_weighted

def check_results(rep, non_rep, out_file_dir=None, desc_str=None):
    """
    Deprecated:
    Outputs results in readable text file table format

    Parameters
    ----------
    rep : dict(str, dict)
        Key: Combined comparison name (LHH0048_LHH0054L)
        Value: dict containing keys: 'emd_value' and/or 'j_value'
        Dictionary containing information for replicate comparisons
    non_rep : dict(str, dict)
        Key: Combined comparison name (LHH0048_LHH0054L)
        Value: dict containing keys: 'emd_value' and/or 'j_value'
        Dictionary containing information for non-replicate comparisons
    out_file_dir : str, optional
        Default is None
    desc_str : str, optional
        Used as file name to describe settings/parameters for this comparison
        Not optional if out_file_dir is not None
        Default is None
    """

    log.info(f"Number of known replicates: {len(rep)}")
    log.info(f"Number of non-replicates or unknown: {len(non_rep)}")

    for value_type in ['emd_value', 'j_value']:
        out_file = None
        out_file_path = None
        if out_file_dir:
            if not os.path.isdir(out_file_dir):
                os.mkdir(out_file_dir)

            if desc_str is None:
                out_file_path = os.path.join(out_file_dir, f'{value_type}.txt')
            else:
                out_file_path = os.path.join(out_file_dir,
                                             f'{desc_str}.{value_type}.txt')

            out_file = open(out_file_path, 'w')

        rep_table = PrettyTable(['Comparison', 'emd_value', 'j_value'])
        rep_table.sortby = value_type
        rep_table.reversesort = True

        for comparison_value in [rep, non_rep]:
            if len(comparison_value) == 0:
                continue

            for k, value_dict in comparison_value.items():
                rep_table.add_row([k, round(value_dict['emd_value'], 5),
                                   round(value_dict['j_value'], 5)])

            if comparison_value == rep:
                temp_str = f'Replicates sorted by {value_type}'
            else:
                temp_str = f'Non-Replicates sorted by {value_type}'
            temp_str += f'\n{rep_table.get_string()}'
            if out_file:
                out_file.write(temp_str + '\n')
            log.info(temp_str)
            rep_table.clear_rows()

        replicate_values = [x[value_type] for x in rep.values()]
        non_replicate_values = [x[value_type] for x in non_rep.values()]

        # No more statistics can be made without knowing replicates
        if len(non_replicate_values) == 0 or len(replicate_values) == 0:
            return

        min_diff = np.min(replicate_values) - np.max(non_replicate_values)
        avg_diff = np.mean(replicate_values) - np.mean(non_replicate_values)
        min_rep = np.min(replicate_values)
        max_non_rep = np.max(non_replicate_values)
        temp_str = f"Min replicate value: " \
                   f"{min(rep, key=lambda x: rep[x][value_type])} -> {min_rep}\n" \
                   f"Max non-replicate value: " \
                   f"{max(non_rep, key=lambda x: non_rep[x][value_type])} -> {max_non_rep}\n" \
                   f"Min diff between replicates and non-replicates: {min_diff}\n" \
                   f"Diff between replicate and non-replicate average: {avg_diff}"
        log.info(temp_str)

        if out_file_path:
            out_file.write(temp_str + '\n')
            log.info(f"Results have been written to {out_file_path}")
            out_file.close()


def parse_genomic_location(
    genomic_location: str,
    chrom_size_file: str,
    bin_size: int
) -> Union[Tuple[str, int, int], None]:
    """
    Parses a genomic region into the single window that is compared instead of the
    sliding windows of every chromosome

    The window is the smallest run of whole bins that covers the region: the start is
    rounded down and the end is rounded up to a multiple of `bin_size`. The region must
    span at least MIN_REGION_SPAN (20 kb), and the window must have MIN_REGION_BINS to
    MAX_REGION_BINS bins, i.e. an adjacency matrix between 3 x 3 and 1000 x 1000

    Parameters
    ----------
    genomic_location : str
        'all' for the sliding windows of the whole genome, or a region 'chrom:start-end'
        such as 'chr5:3000000-5000000'. start and end are in bp and form the half open
        interval [start, end), as in a BED file. Commas are allowed ('chr5:3,000,000-5,000,000')
        and the chromosome may be given with or without the 'chr' prefix ('5:3000000-5000000')
    chrom_size_file : str
        Chromosome size file, which may name the chromosome with or without the 'chr' prefix
    bin_size : int
        Binning resolution

    Returns
    -------
    None if `genomic_location` is 'all', otherwise (chrom, window_start, window_end)
    where chrom is standardized to 'chr*' (e.g. 'chr5')

    Raises
    ------
    ValueError
        If the region is malformed, is not within the chromosome or breaks one of the limits
    """
    if genomic_location is None or genomic_location == 'all':
        return None

    match = GENOMIC_LOCATION_RE.fullmatch(genomic_location.strip())
    if match is None:
        log.error(f'Invalid genomic_location "{genomic_location}": expected "chrom:start-end", '
                  f'e.g. "chr5:3000000-5000000"')
        raise ValueError(f'Invalid genomic_location "{genomic_location}": expected "chrom:start-end", '
                         f'e.g. "chr5:3000000-5000000"')

    chrom = standardize_chrom_name(match.group(1))
    start, end = (int(x.replace(',', '')) for x in match.group(2, 3))

    # Smallest run of whole bins covering [start, end)
    window_start = math.floor(start / bin_size) * bin_size
    window_end = math.ceil(end / bin_size) * bin_size
    num_bins = (window_end - window_start) // bin_size

    chrom_sizes = read_chrom_sizes(chrom_size_file)
    size_chrom_name = check_alt_chrom_name(chrom, list(chrom_sizes))

    if size_chrom_name is None:
        error = f'{chrom} (or {chrom[3:]}) not found in {chrom_size_file}'
    elif not start < end <= chrom_sizes[size_chrom_name]:
        error = f'expected start < end <= {chrom_sizes[size_chrom_name]}, the size of {chrom}'
    elif end - start < MIN_REGION_SPAN:
        error = f'the region spans {end - start} bp, less than the minimum of {MIN_REGION_SPAN} bp'
    elif not MIN_REGION_BINS <= num_bins <= MAX_REGION_BINS:
        error = (f'the window {chrom}:{window_start}-{window_end} covering the region has {num_bins} bins '
                 f'of {bin_size} bp, outside the allowed {MIN_REGION_BINS} to {MAX_REGION_BINS}. '
                 f'Change the region or the bin size')
    else:
        return chrom, window_start, window_end

    log.error(f'Invalid genomic_location "{genomic_location}": {error}')
    raise ValueError(f'Invalid genomic_location "{genomic_location}": {error}')

def read_data(
    input_data_file: str,
    chrom_size_file: str,
    chroms_to_load: List[str] = None,
    window_size: int = 3000000,
    window_stride: int = 2,
    num_peaks: int = None,
    base_chrom: str = 'chr1',
    normalization: str = 'NONE',
    bin_size: int = 1,
    output_dir: str = 'output',
    min_hic_value: int = 1,
    min_bedgraph_value: int = 1,
    ba_mult: int = 1,
    coverage_weighting: str = 'mult',
    ablation: str = 'None',
    ablation_hic_10x_rep2: bool = False,
    fixed_seed: bool = False,
    genomic_location: str = 'all',
    gamma: float = None,
    shapley: bool = False
) -> Dict[str, GenomeBinData]:
    """
    Reads all samples that are found in loop_data_dir.

    loop_data_dir/peak_data_dir/bedgraph_data_dir do not have to be separate
    directories.

    Parameters
    ----------
    input_data_file : str
        File with file paths to all necessary input files.
        Format:
        sample1_name bedgraph1_file   hic1_file
        sample2_name bedgraph2_file   hic2_file
        ...
    chrom_size_file : str
        Path to chromosome size file
    chroms_to_load : list, optional
        Specify specific chromosomes to load instead of the entire genome
    window_size : int
        Size of sliding window
    window_stride : int
        Factor of window_size to stride (e.g. 2 means stride by window_size/2)
    num_peaks : int, optional
        Number of peaks to keep in `base_chrom`
        The same ratio of peaks will be kept for other chromosomes
        If None, keep all peaks
    base_chrom : str
        Chromosome to use when selecting number of peaks
    normalization : str, optional
        Hi-C normalization method. 
        Options are same as .hic file normalization methods (e.g. NONE, VC, VC_SQRT, KR)
        WARNING: If the normalization results in values that are less than 1,
        then those values will be filtered out by two mechanisms:
        (1) min_hic_value
        (2) chrom_bin_data.py: `laplacian` function which considers values ≥ 1
    bin_size : int
        Binning resolution
    min_hic_value : int
        Minimum Hi-C value to consider a valid interaction
    min_bedgraph_value : int
        Minimum bedgraph value to consider a valid binding affinity
    output_dir : str
        Directory to output data
    ba_mult : int
        Multiplicative factor to multiply binding affinity values by
        This is recommended if there are binding affinity values are less than 1
        (i.e. non-integer values between 0 and 1)
    genomic_location : str
        'all' (default) for the sliding windows of the whole genome, or a genomic region
        'chrom:start-end' in bp such as 'chr5:3000000-5000000' (see `parse_genomic_location`).
        A region is compared as a single window instead: the smallest run of whole bins that
        covers it, which must hold 3 to 1000 bins, from a region spanning at least 20 kb.
        The chromosome may be named with or without the 'chr' prefix, in the region and in
        each input file, and is named 'chr*' in the outputs. `window_size`, `window_stride`
        and `chroms_to_load` are then ignored. Not implemented for the shuffle ablations
        or with peak filtering
    shapley : bool
        Whether to also build the structure only and binding only graphs of every window
        that compare(..., shapley=True) needs for the Shapley attribution. Roughly doubles
        the memory taken by the adjacency matrices. Not implemented with peak filtering

    Returns
    -------
    OrderedDict[str, GenomeBinData]
    """
    total_start_time = time.time()
    os.makedirs(f'{output_dir}/timings', exist_ok=True)
    sample_data_dict = OrderedDict()

    if not os.path.isfile(chrom_size_file):
        log.error(f"Chrom size file: {chrom_size_file} is not a valid file")
        return sample_data_dict

    if not os.path.isfile(input_data_file):
        log.error(f"Data file: {input_data_file} is not a valid file")
        return sample_data_dict

    # Single window instead of the sliding windows of the genome
    region = parse_genomic_location(genomic_location, chrom_size_file, bin_size)
    if region is not None:
        region_chrom, window_start, window_end = region
        log.info(f'genomic_location "{genomic_location}": comparing the single window '
                 f'{region_chrom}:{window_start}-{window_end} ({(window_end - window_start) // bin_size} '
                 f'bins of {bin_size} bp) instead of sliding windows')
        log.warning('Window scores depend on the size of the window (N x N matrices), so compare the '
                    'scores of a genomic_location only with scores of the same region and bin_size, '
                    'not with genome-wide scores')

        if chroms_to_load:
            log.warning(f'chroms_to_load {chroms_to_load} is ignored for a genomic_location')

        if ablation in ["Shuffle Binding", "Shuffle Structure", "Shuffle Both"]:
            log.error(f'Ablation "{ablation}" shuffles the windows of a chromosome, which needs more '
                      f'than the single window of a genomic_location')
            raise ValueError(f'Ablation "{ablation}" shuffles the windows of a chromosome, which needs more '
                             f'than the single window of a genomic_location')

    # Get input file names
    input_sample_files = []
    num_files = 3
    with open(input_data_file) as in_file:
        for line in in_file:
            sample_files = line.split()
            if len(sample_files) not in [3, 4]:
                log.error(f"Invalid number of columns in {input_data_file}")
                return sample_data_dict
            if len(sample_files) == 4:
                num_files = 4
            input_sample_files.append(sample_files)

    # Peak filtering changes the default graphs in `preprocess` but not the Shapley variants
    if shapley and (num_files == 4 or num_peaks is not None):
        log.error('The Shapley attribution is not implemented together with peak filtering')
        raise NotImplementedError('The Shapley attribution is not implemented together with peak filtering')

    # Peaks are thresholded by the ratio kept in base_chrom, which a single region does not have
    if region is not None and (num_files == 4 or num_peaks is not None):
        log.error('Peak filtering is not implemented for a genomic_location')
        raise NotImplementedError('Peak filtering is not implemented for a genomic_location')

    # Dict from sample name to an integer for random seed
    if not fixed_seed:
        # Variable seed per sample
        rng_seed_dict = {sample_files[0] : i for i, sample_files in enumerate(input_sample_files)} 
    else:
        # Fixed seed per sample
        rng_seed_dict = {sample_files[0] : 42 for sample_files in input_sample_files}

    if ablation in ["Shuffle Binding", "Shuffle Structure", "Shuffle Both"]:
        # For this ablation, we want variable shuffling for each sample
        rng_seed_dict = {sample_files[0] : i for i, sample_files in enumerate(input_sample_files)}

        # We want to shuffle all samples
        # It should *suffice* to shuffle just the "left" of a pair
        # but it is suitable to also shuffle both of a pair (i.e. both)
        to_shuffle = True
    else:
        to_shuffle = False


    sample_timings = OrderedDict() 
    for sample_files in input_sample_files:
        sample_start_time = time.time()

        sample_name = sample_files[0]
        bedgraph_file = sample_files[1] # binding affinity
        chrom_structure_file = sample_files[2] # Hi-C or bedpe
        peak_file = sample_files[3] if num_files == 4 else None

        # Check for file validity
        invalid_file = False
        for i in range(1, num_files):
            if not os.path.isfile(sample_files[i]):
                log.error(f"Data file: {sample_files[i]} is not a valid file")
                invalid_file = True
                break
        if invalid_file:
            continue

        log.info(f'Loading {sample_name} ...')

        peak_dict = None 
        from_peak_file = True
        if peak_file:
            peak_dict = read_peak_file(peak_file)
        elif num_peaks is not None:
            # peak_dict = generate_peak_file(bin_size, chrom_size_file)
            peak_dict = construct_empty_peak_dict(chrom_size_file)
            from_peak_file = False

        if to_shuffle:
            log.info(f'Shuffling \'{ablation}\' of {sample_name} ...')
        
        gld = GenomeBinData(chrom_size_file, bedgraph_file, chrom_structure_file,
                            window_size, window_stride, normalization,
                            peak_dict=peak_dict, from_peak_file=from_peak_file,
                            base_chrom=base_chrom, num_peaks=num_peaks,
                            chroms_to_load=chroms_to_load, bin_size=bin_size,
                            min_hic_value=min_hic_value, min_bedgraph_value=min_bedgraph_value,
                            rng_seed=rng_seed_dict[sample_name], ba_mult=ba_mult, output_dir=output_dir, coverage_weighting=coverage_weighting, 
                            ablation=ablation, ablation_hic_10x_rep2=ablation_hic_10x_rep2, to_shuffle=to_shuffle, gamma=gamma,
                            shapley=shapley, region=region)

        sample_data_dict[sample_name] = gld
        sample_timings[sample_name] = time.time() - sample_start_time

    with open(f'{output_dir}/timings/read_data.txt', 'w') as out_file:
        out_file.write(f'sample_name\ttime_taken\n')
        for sample_name, sample_timing in sample_timings.items():
            out_file.write(f'{sample_name}\t{sample_timing}\n')
        out_file.write(f'total\t{time.time() - total_start_time}\n')

    return sample_data_dict


def preprocess(
    sample_dict: OrderedDict,
    both_peak_support: bool = False,
    output_dir: str = 'output',
    diagnostic_plots: bool = True
) -> None:
    """
    Filters out Hi-C contacts without peak support using matrix operations

    Removes chromosomes that result in all zero matrices after filtering

    Optionally saves diagnostic plots of peak array

    Parameters
    ----------
    sample_dict : OrderedDict[str, GenomeLoopData]
        Samples to compare
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
    for sample_data in sample_dict.values():
        sample_data.preprocess(both_peak_support=both_peak_support,
                               output_dir=output_dir, 
                               diagnostic_plots=diagnostic_plots)


def read_peak_file(
    peak_file_path: str
) -> Dict[str, list]:
    """
    Finds the start and ends of every peak in chromosome for one sample. Find
    the max value within the interval for each peak using the bedgraph file.

    File format must have at least 3 columns with the first 3 being:
    chrom_name   start   end

    Parameters
    ----------
    peak_file_path : str
        File path of peak file

    Returns
    -------
    dict[str, list]
        Dictionary containing list of peaks start and ends for every chromosome
    """
    peak_dict = {}

    with open(peak_file_path) as peak_file:
        for line in peak_file:
            data = line.split()
            chrom_name = data[0]

            try:
                peak_start = int(data[1])
                peak_end = int(data[2])
            except ValueError:  # Sometimes have peaks with 1+E08 as a value
                log.error(f'Invalid peak: {line}')
                continue

            if chrom_name not in peak_dict:
                peak_dict[chrom_name] = []

            peak_dict[chrom_name].append([peak_start, peak_end,
                                          peak_end - peak_start])
    return peak_dict


# def generate_peak_file(
#         bin_size: int,
#         chrom_size_file: str,
# ) -> Dict[str, list]:
#     """
#     Generates peak dictionary of same format as read_peak_file with
#     evenly spaced peaks across all chromosomes

#     This function calls pybedtools window_maker which is a wrapper for 
#     bedtools makewindows to construct this peak dictionary

#     Parameters
#     ----------
#     bin_size : int
#         The resolution or window size 
#     chrom_size_file : str
#         Chromosome size file
    
#     Returns
#     -------
#     dict[str, list]
#         Dictionary containing list of peaks start and ends for every chromosome
#     """
#     peak_dict = {}
    
#     # Equivalent to: bedtools makewindows -g chrom_size_file -w bin_size
#     windows = pybedtools.BedTool().window_maker(g=chrom_size_file, w=bin_size)

#     for feature in windows:
#         chrom = feature.chrom
#         start = feature.start
#         end = feature.end
        
#         if chrom not in peak_dict:
#             peak_dict[chrom] = []
            
#         peak_dict[chrom].append([start, end, 
#                                  end - start])
        
#     return peak_dict

def construct_empty_peak_dict(
        chrom_size_file: str,
) -> Dict[str, list]:
    """
    Constructs an empty peak dictionary of same format as read_peak_file with
    no peaks for any chromosome

    Parameters
    ----------
    chrom_size_file : str
        Chromosome size file
    
    Returns
    -------
    dict[str, list]
        Dictionary containing empty list of peaks for every chromosome
    """
    peak_dict = {}
    
    with open(chrom_size_file) as in_file:
        for line in in_file:
            line = line.strip().split()
            chrom_name = line[0]
            peak_dict[chrom_name] = []
        
    return peak_dict
