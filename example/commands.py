import itertools
import os
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument('sample_list_file', type=click.File('r'))
@click.argument('output_pair_file', type=click.File('w'))
def make_pairs(sample_list_file, output_pair_file):
    sample_list = [sample.strip() for sample in sample_list_file]
    for pair in itertools.combinations(sample_list, 2):
        line = "\t".join(pair) + '\n'
        output_pair_file.write(line)


@cli.command()
@click.argument('sample_list_file', type=click.File('r'))
@click.argument('sample_input_file', type=click.File('w'))
@click.argument('sample_data_dir')
@click.option('-p', '--use_peaks', default=False, type=bool, deprecated=True)
@click.option('-b', '--use_bedpe', default=False, type=bool) # Default is False i.e. default to Hi-C format
def make_sample_input_file(sample_list_file, sample_input_file, sample_data_dir, use_peaks, use_bedpe):
    bg_ext = '.bedgraph'
    bw_ext = '.bigWig'
    if use_bedpe:
        chrom_ext1 = '.bedpe'
        chrom_ext2 = '.hic' # takes less precedence
    else:
        chrom_ext1 = '.hic'
        chrom_ext2 = '.bedpe' # takes less precedence
    peak_ext = '.bed'

    for sample_name in sample_list_file:
        sample_name = sample_name.strip()

        # Get file paths for each needed file
        bg_file_path = None
        hic_file_path = None
        peak_file_path = None
        for file in os.scandir(sample_data_dir):
            if file.name.lower().endswith(bg_ext) and sample_name.lower() in file.name.lower():
                bg_file_path = file.path
            elif (file.name.lower().endswith(chrom_ext1)) and sample_name.lower() in file.name.lower():
                hic_file_path = file.path
            elif use_peaks and file.name.lower().endswith(peak_ext) and sample_name.lower() in file.name.lower():
                peak_file_path = file.path

        # If chrom_ext1 wasn't found after scanning ALL files, scan again for chrom_ext2
        if not hic_file_path:
            print(f"Couldn't find chromatin structure file {chrom_ext1} for {sample_name}, trying {chrom_ext2}..."
                  "To disable this behaviour, please do not include this sample in the sample list file.")
            for file in os.scandir(sample_data_dir):
                if (file.name.lower().endswith(chrom_ext2)) and sample_name.lower() in file.name.lower():
                    hic_file_path = file.path
                    break

        if not bg_file_path:
            print(f"Couldn't find binding affinity file {bg_ext} for {sample_name}, trying {bw_ext}..."
                  "To disable this behaviour, please do not include this sample in the sample list file.")
            for file in os.scandir(sample_data_dir):
                if (file.name.lower().endswith(bw_ext)) and sample_name.lower() in file.name.lower():
                    bg_file_path = file.path
                    break

        if not bg_file_path or not hic_file_path:
            print(f'Missing files for {sample_name}. Skipping.')
            continue

        if use_peaks and not peak_file_path:
            print(f'Missing peak file for {sample_name} but use_peaks is True. Skipping.')
            continue

        if use_peaks:
            sample_input_file.write(f'{sample_name}\t{bg_file_path}\t{hic_file_path}\t{peak_file_path}\n')
            # A difference from the original program is that the last column is the peak
            # The oriignal code had it as the second to last column
        else:
            sample_input_file.write(f'{sample_name}\t{bg_file_path}\t{hic_file_path}\n')


if __name__ == '__main__':
    cli()
