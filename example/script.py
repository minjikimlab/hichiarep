import sys
import click
from logging.config import fileConfig
import os

# Add parent folder to PATH if chia_rep is not installed
sys.path.append('..')
import hichiarep

# To allow logging INFO level statements to stdout
fileConfig('log.conf')
# script_dir = os.path.dirname(os.path.abspath(__file__))
# log_conf_path = os.path.join(script_dir, 'log.conf')
# fileConfig(log_conf_path)
# 
# def parse_density_method(ctx, param, value):
#     '''
#     Deprecated
#     '''
#     try:
#         # Should automatically parse the string "0.8" to float 0.8 e.t.c
#         return float(value)
#     except ValueError:
#         # Otherwise, just return the string e.g. "L2"
#         return value
    
# @click.option('-d', '--density-method', default='L2', callback=parse_density_method)

def parse_float(ctx, param, value):
    try:
        # Should automatically parse the string "0.8" to float 0.8 e.t.c
        return float(value)
    except ValueError:
        # Otherwise, just return the string e.g. "L2"
        return value

@click.command()
@click.version_option(hichiarep.__version__)
@click.argument('input_data_file', type=click.Path(exists=True))
@click.argument('chrom_size_file', type=click.Path(exists=True))
@click.argument('compare_list_file', type=click.Path(exists=True))
@click.argument('window_size', default=5000000, type=int)
@click.argument('bin_size', default=10000, type=int)
@click.argument('chroms_to_load', nargs=-1)
@click.option('-l', '--min-hic-value', default=1, type=int)
@click.option('-b', '--min-bedgraph-value', default=1, type=int)
@click.option('-g', '--do-output-graph', default=False, type=bool)
@click.option('-o', '--output-dir', default='output', type=str)
@click.option('-w', '--window-stride', default=2, type=int)
@click.option('-p', '--num-peaks', default=None, type=int, deprecated=True)
@click.option('--normalization', default='NONE', type=str)
@click.option('--method', default='random_walk', type=str)
@click.option('--mu', default=5, type=float)
@click.option('--compare-method', default='spearman', type=str)
@click.option('--cross', default=False, type=bool)
@click.option('--ba-mult', default=1, type=int)
@click.option('--ssp', default=None, type=float)
@click.option('--num-cores', default=1, type=int)
@click.option('--diagnostic-plots', default=True, type=bool)
def main(input_data_file, chrom_size_file, compare_list_file, window_size, bin_size, chroms_to_load,
         min_hic_value, min_bedgraph_value, do_output_graph, output_dir, window_stride, num_peaks, 
         normalization, method, mu, compare_method, cross, ba_mult, ssp, num_cores, diagnostic_plots):
    click.echo(f"Running HiChIA-Rep v{hichiarep.__version__}")
    
    if 'all' in chroms_to_load or len(chroms_to_load) == 0:
        chroms_to_load = None

    param_str = f'{window_size}.{bin_size}.{normalization}.{method}.{mu}.{compare_method}.{cross}.{ba_mult}.{ssp}'

    sample_data_dict = hichiarep.read_data(input_data_file, chrom_size_file, chroms_to_load,
                                          window_size=window_size, window_stride=window_stride, normalization=normalization,
                                          bin_size=bin_size, output_dir=output_dir, num_peaks=num_peaks, base_chrom='chr1',
                                          min_hic_value=min_hic_value, min_bedgraph_value=min_bedgraph_value, ba_mult=ba_mult)
    
    hichiarep.preprocess(sample_data_dict, both_peak_support=False, 
                        output_dir=output_dir, diagnostic_plots=diagnostic_plots)
    
    scores_unweighted, _ = hichiarep.compare(sample_data_dict, 
                              compare_list_file=compare_list_file, method=method, 
                              do_output_graph=do_output_graph, output_dir=output_dir, 
                              mu=mu, compare_method=compare_method, cross=cross,
                              ssp=ssp,
                              num_cores=num_cores, param_str=param_str)
    
    hichiarep.output_to_csv(None, scores_unweighted,
                           output_dir=output_dir, param_str=param_str)

if __name__ == '__main__':
    main()
