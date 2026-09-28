try:
    from importlib.metadata import version
    __version__ = version("HiChIA-Rep")
except Exception:
    __version__ = "unknown"
    
from .chia_rep import read_data, compare, output_to_csv, preprocess, parse_genomic_location
# from .chrom_loop_data import *
# from .genome_loop_data import *
# from .chrom_bin_data import *
# from .genome_bin_data import *