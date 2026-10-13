# HiChIA-Rep
A package for quantifying the similarity between chromatin interactions data enriched for protein binding sites or open chromatin regions (e.g. ChIA-PET, HiChIP, ChIATAC, HiCAR and related data).

    
## Methods Overview 
### Data

**Chromatin Structure**\
The chromatin structure is assumed to be a `.hic` file (with normalization="NONE" available) or a `.bedpe` file that is commonly used to store the "loops" of processed ChIA-PET experiments. The toggle between these two files is available as the parameter `use_bedpe` in the `commands.py make-sample-input-file` program. The program bins the chromatin structure data at a specified resolution into an $M \times M$ adjacency matrix with $M$ being the number of nodes, that is the number of bins in a given window. The adjacency matrix $A$ should contain *integers* where $A_{i,j}$ is the number of inter-ligated fragments captured between binned genomic loci $i$ and $j$. This matrix is symmetric and non-negative and may contain isolated nodes due to centromeres and telomeres. 

**Enrichment Signal**\
The enrichment signal data is assumed to be a `.bedgraph` or `.bigWig` file and represents either the protein binding (e.g. ChIA-PET) or DNA accessibility (e.g. ChIATAC). The program bins the binding affinity data at a specified resolution into an $M$-dimensional vector with $M$ being the number of nodes, that is the number of bins in a given window. This vector $b$ should similarly contain *integers* where $b_i$ is the number of captured fragments aligning to binned genomic locus $i$. This vector is non-negative. If the values are not integer counts and contain floating point values between 0 and 1, then it is recommended to multiply the binding affinity values by a large fixed constant, which can be specified via the `ba_mult` parameter. 

### Graph signal processing
HiChIA-Rep uses graph signal processing techniques to compare a sliding window between two input samples. Specifically, we view the enrichment signal $b$ (e.g. protein binding) to be a 1D signal defined over the nodes of the chromatin graph, whose edge weights are assigned from the chromatin contact matrix $A$. From here, either random walk or diffusion is performed on the enrichment signal to propogate structural information from the graph into the enrichment signal. This step requires a key parameter $\mu$, which either specifies the number of random walks or the time step in the diffusion process. The processed enrichment signals after random walk or diffusion are then compared using either the Spearman correlation or the Jensen Shannon Divergence (JSD), transformed into a similarity measure to lie in [-1, 1]. 

This procedure is repeated for each sliding window independently, and so HiChIA-Rep can quantify the similarity on a per-window basis across the entire genome. User's may wish to inspect the window scores, in addition to the final reproducibility measure, which simply takes an average of all the window scores. Please see the paper for more details. 

    
## Program overview 

### Installation: 

```bash    
# Install from github
git clone https://github.com/minjikimlab/hichiarep.git    
```


### Dependencies:
Make a conda environment using `environment.yml`:

```
conda env create -f environment.yml
```

The `environment.yml` file should look something like this:
```
name: hichia-env
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.10
  - numpy>=2.2,<2.3
  - scipy>=1.15,<1.16
  - matplotlib>=3.10,<3.11
  - click
  - pybedgraph>=0.5
  - pybedtools>=0.12
  - pybigwig>=0.3.16
  - bedtools
  - sphinx # Just for building docs
  - libcurl # For hic-straw
  - pip
  - pip:
      - hic-straw==1.3
```


### Input metadata files
The main program takes input two metadata files that need to be generated, which specify the data and which combinations of pairs to compare. 

First, ensure that the data is in the same folder (either by making copies or simlinks). Also, ensure that the chromatin structure file has the correct extension (`.hic` | `.bedpe`) and similarly for the enrichment signal file (`.bedgraph` | `.bigWig`).

#### Sample file
The only file that the user needs to explicitly make is the **sample file**. Simply write the file names without the extensions in each line, one for each sample. For instance, if we have files `GM12878_chiatac.hic` and `GM12878_chiatac.bedgraph` and `A549_chiatac.hic` and `A549_chiatac.bedgraph`, then the following is a suitable sample file:

`sample_file.txt`:
```
GM12878_chiatac
A549_chiatac
```

Note that you don't need to write the entire filename and just a *substring* of each filename is sufficient, as long as it is not ambiguous i.e.,

`sample_file.txt`:
```
GM12878
A549
```
is another suitable sample file. 



> #### Example
> In the example data folder `example/example_data`, we have two ChIA-PET samples (GM12878 CTCF rep1 and GM12878 RNAPII rep1):
> 
> ```bash
> [sionkim@gl-login5 example_data]$ ls
> GM12878_ctcf_chiapet_chr22_rep1_cov_ENCFF730GKJ.bedgraph  GM12878_ctcf_chiapet_chr22_rep1_loops_ENCFF780PGS.bedpe GM12878_rnapii_chiapet_chr22_rep1_cov_ENCFF621IJG.bedgraph  GM12878_rnapii_chiapet_chr22_rep1_loops_ENCFF040KUS.bedpe
> ```


In order to compare GM12878 CTCF rep1 and GM12878 RNAPII rep1, a valid sample file is 

`example/sample_list.txt`
```
GM12878_ctcf_chiapet_chr22_rep1
GM12878_rnapii_chiapet_chr22_rep1
```

#### Pairs file
We may have more than 2 samples (n > 2) that we wish to compare. In this case, we may not want to compare all possible combinations of the n files (n choose 2). We specify the pairs which we wish to compare using the **pairs file**. 

If the user wants to compare all possible combinations, we provide a helper function that generates the pairs file automatically. 

```bash
conda activate hichia-env # activate environment
python commands.py make-pairs <sample file> <pairs file>
```

> #### Example
> ```bash
> conda activate hichia-env # activate environment
> python commands.py make-pairs example_sample.txt example_pairs.txt
> ```
> The example generates all possible combinations between GM12878 CTCF rep1 and GM12878 RNAPII rep1, which is trivially just one comparison
> 
> `example_pairs.txt`
> ```bash
> GM12878_ctcf_chiapet_chr22_rep1	GM12878_rnapii_chiapet_chr22_rep1
> ```

#### Sample input file
The sample input file takes the **sample file** and associates it with the data paths, automatically finding the chromatin structure and enrichment signal files via the extension. The command is the following

```bash
conda activate hichia-env # activate environment (if not already activated)

python commands.py make-sample-input-file <sample file> <sample input file> <data directory> \
    --use_bedpe <True or False> 
```
where the `--use_bedpe` parameter specifies to prioritize finding Hi-C (.hic) files over .bedpe files for the chromatin structure. 

> #### Example
> In the example, since we have .bedpe files we will set the `--use_bedpe True`.
> ```bash
> conda activate hichia-env # activate environment
> python commands.py make-sample-input-file example_sample.txt example_sample_input_file.txt \
>    /nfs/turbo/umms-minjilab/sionkim/chia_rep/example/example_data \
>    --use_bedpe True 
> ```
> `example_sample_input_file.txt`
> ```bash
> GM12878_ctcf_chiapet_chr22_rep1	/nfs/turbo/umms-minjilab/sionkim/chia_rep/example/example_data/GM12878_ctcf_chiapet_chr22_rep1_cov_ENCFF730GKJ.bedgraph	/nfs/turbo/umms-minjilab/sionkim/chia_rep/example/example_data/GM12878_ctcf_chiapet_chr22_rep1_loops_ENCFF780PGS.bedpe
> GM12878_rnapii_chiapet_chr22_rep1	/nfs/turbo/umms-minjilab/sionkim/chia_rep/example/example_data/GM12878_rnapii_chiapet_chr22_rep1_cov_ENCFF621IJG.bedgraph	/nfs/turbo/umms-minjilab/sionkim/chia_rep/example/example_data/GM12878_rnapii_chiapet_chr22_rep1_loops_ENCFF040KUS.bedpe
> ```

The main program will then take the generated **pairs file** and **sample input file** as inputs.


### Running main program

#### Parameters

| Parameter | Options (Default) | Description |
|-----------|-------------------|-------------|
| `input_data_file` | Path (required) | Path to **sample input file** containing file paths to bedgraph and chromatin structure files (see [sample file section](#input-metadata-files))
| `chrom_size_file` | Path (required) | Path to chromosome size file (e.g., hg38.chrom.sizes) |
| `compare_list_file` | Path (required) | Path to pairs file specifying which samples to compare |
| `window_size` | int (5000000) | Size of sliding window in base pairs |
| `bin_size` | int (10000) | Binning resolution in base pairs |
| `chroms_to_load` | str... (all) | Chromosomes to analyze; use "all" or leave empty for all chromosomes |
| `-o`, `--output-dir` | str ("output") | Directory to output results |
| `-w`, `--window-stride` | int (2) | Stride factor (e.g., 2 means stride by window_size/2) |
| `--method` | str ("random_walk") | Comparison method: "random_walk" or "diffusion" |
| `--mu` | float (5) | Random walk steps or diffusion time parameter |
| `--compare-method` | str ("spearman") | Signal comparison method: "spearman", "pearson", "jsd" or "OT" (earth mover's distance, needs `pip install POT`) |
| `--ba-mult` | int (1) | Multiplier for binding affinity values (use if values are <1) |
| `--num-cores` | int (1) | Number of cores for parallel processing |
| `--diagnostic-plots` | bool (True) | Save diagnostic plots of peak arrays |
| `--min-hic-value` | int (1) | Minimum Hi-C value to consider a valid interaction |
| `--min-bedgraph-value` | int (1) | Minimum bedgraph value to consider valid binding affinity |
| `--do-output-graph` | bool (False) | Output processed .npy graphs for chr1 windows (debugging) |
| `--genomic-location` | str ("all") | Compare a single region `chrom:start-end` (bp) instead of sliding windows over the genome (see [Comparing a single genomic region](#comparing-a-single-genomic-region)) |
| `--shapley` | bool (False) | Also score each window with structure only (`W_S`) and binding only (`W_B`), and report `alpha_shapley`, the share of the window score attributed to structure. `random_walk` only, roughly triples the comparison time |
| `--ablation` | str ("None") | Ablate the chromatin structure or enrichment signal: "None", "Uniform", "Uniform Random", "Permute Binding", "Identity", "Uniform Structure", "Expected", "Permute Structure", "Shuffle Binding", "Shuffle Structure" or "Shuffle Both" (for ablation studies) |
| `--fixed-seed` | bool (False) | Use the same random seed for every sample in the randomized ablations instead of one seed per sample |

It is recommended that the window size sufficiently captures important structures that you wish to compare. The window size is measured from the main diagonal and is by default 5,000,000 i.e. 5 Mb. If you wish to increase window size, such as 10 Mb, then it is recommended to keep the bin_size ≥ 10000 (at least 10 kb). 

It is recommended to keep mu (i.e. $\mu$) relatively small. If the method is random walk, then it is suggested to keep mu ≤ 5. 

#### Main program

Here, we provide some example calls of the main program. 

```bash
conda activate hichia-env # activate environment (if not already activated)
```

Default

```bash
python script.py example_sample_input_file.txt hg38.chrom.sizes example_pairs.txt 
```

Specify some parameters 

```bash
python script.py example_sample_input_file.txt hg38.chrom.sizes example_pairs.txt 5000000 10000 all
```


Specify many parameters

```bash
python script.py example_sample_input_file.txt hg38.chrom.sizes example_pairs.txt 5000000 10000 all \
    --window-stride 2 \
    --do-output-graph False \
    --output-dir example_output \
    --normalization NONE \
    --method random_walk \
    --mu 5 \
    --num-cores 4 \
    --compare-method spearman \
    --diagnostic-plots True
```

#### Comparing a single genomic region

HiChIA-Rep can compare a single genomic region (as opposed to a sliding window) via `--genomic-location chrom:start-end`

> ### Example
> ```bash
> python script.py example_sample_input_file.txt hg38.chrom.sizes example_pairs.txt 5000000 10000 all \
>     --genomic-location chr13:10000000-25000000
> ```

- `start`, `end` is half open genomic interval [start, end) (e.g. BED).
- The genomic span is restricted to be ≥20 kb between 10 and 1000 bins (e.g. between 100kb and 10Mb at 10kb bin_size)
- `window_size`, `--window-stride`, `chroms_to_load`, `--num-cores` are deactivated.
- The window score is expected to be dependent with the number of bins, so it is recommended to compare with scores of the same number of bins (number of bins = genomic span / bin size).


## Results
The results are contained in a folder where the main program is run, titled as `output` (default) or specified via the `--output-dir` parameter. 

The two main outputs are 
1. Score matrices (.csv): `unweighted_scores.csv`, the average of the window scores, and `beta_weighted_scores.csv`, the average weighted by `beta`, the fraction of bins of a window that are live in both samples
2. Window scores (.txt)

> ### Example
> `example/example_output/5000000.10000.NONE.random_walk.5.0.spearman.False.1.None.mult.None.None.live-either.minlive-10/scores/unweighted_scores.csv`
> 
> | Sample Name | GM12878_ctcf_chiapet_chr22_rep1 | GM12878_rnapii_chiapet_chr22_rep1 |
> |-------------|----------------------------------|-----------------------------------|
> | GM12878_ctcf_chiapet_chr22_rep1 | 0 | 0.4116991970655355 |
> | GM12878_rnapii_chiapet_chr22_rep1 | 0.4116991970655355 | 0 |
> 
> `example/example_output/5000000.10000.NONE.random_walk.5.0.spearman.False.1.None.mult.None.None.live-either.minlive-10/scores/windows/GM12878_ctcf_chiapet_chr22_rep1_loops_ENCFF780PGS_GM12878_rnapii_chiapet_chr22_rep1_loops_ENCFF040KUS_chr22.txt`
> 
> ```
> chrom_name	window_start	window_end	random_walk_score	beta	beta_pmf
> chr22	0	5000000	nan	0.0	0.0
> chr22	2500000	7500000	nan	0.0	0.0
> chr22	5000000	10000000	nan	0.0	0.0
> chr22	7500000	12500000	0.0	0.0	0.0
> chr22	10000000	15000000	0.0	0.0	0.0
> chr22	12500000	17500000	0.42784323705824023	0.066	0.009184525466184246
> chr22	15000000	20000000	0.4892498292849478	0.27	0.03757305872529919
> chr22	17500000	22500000	0.39108467593088747	0.432	0.0601168939604787
> chr22	20000000	25000000	0.38213041314086615	0.49	0.06818814361258001
> chr22	22500000	27500000	0.4577776047041415	0.462	0.06429167826328973
> chr22	25000000	30000000	0.4340993918777546	0.522	0.07264124686891177
> chr22	27500000	32500000	0.5082767020329111	0.708	0.09852490954634009
> ...
> ```
> 
> With `--shapley True`, the window scores gain the columns `W_S`, `W_B` and `alpha_shapley`.









## Contact
Contact Minji (minjilab@umich.edu) for general questions or Sion (sionkim@umich.edu). 
