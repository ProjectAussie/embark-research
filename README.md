# embark-research
Repository of code used in Embark publications.


# Table of Contents
## 2020
### Sams, Aaron J, Ford, Brett, Gardner, Adam, and Boyko, Adam R. "Examination of the efficacy of small genetic panels in genomic conservation of companion animal populations." Evolutionary Applications. (In revision - ADD METADATA WHEN PUBLISHED)

* `analyze_data_and_make_figures_and_tables.py` - A python (3) (we used 3.7.4 at runtime) script to aggregate and analyze raw data from simulation outputs.
* `requirements.txt` - List of versions of non-standard library python packages used in `analyze_data_and_make_figures_and_tables.py`
* `diversity_matechoice_template_for_manuscript.slim` - A template for SLIM 3 used to generate all simulations in this manuscript.

To generate summary stats and build figures and tables (replicate data in manuscript)...
```py
python analyze_data_and_make_figures_and_tables.py \
    --output-root-directory {path_to_output_directory} \
    --simulation-results-root-directory {path_to_unzipped_raw_data} \
    --get-data-from-results-files
```

Supplementary materials are available with the [online version of this manuscript]() # will add when final version is available
Raw simulation data is available on [DRYAD]()

## 2019
### Sams, Aaron J, and Adam R Boyko. “Fine-Scale Resolution of Runs of Homozygosity Reveal Patterns of Inbreeding and Substantial Overlap with Recessive Disease Genotypes in Domestic Dogs.” G3: Genes|Genomes|Genetics, November 14, 2018, g3.200836.2018–7. doi:10.1534/g3.118.200836.

* `postprocess_germline.py` - A python script with post-processing algorithm for filling gaps and filtering on marker counts in merged homozygosity tracts from germline (1.5.1) using --bits 1.

All other supplementary materials are available at [figshare](https://figshare.com/articles/Supplementary_Material_for_Sams_and_Boyko_2018/7330151)