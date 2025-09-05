#!/usr/bin/env python3

import process_data
import combined_analysis
from create_venn_diagram import create_venn_diagram
from taxa_barplot_heatmap import create_taxa_barplot
from network_plots import create_network_plots
from boxplots import create_taxa_boxplot, create_alpha_diversity_boxplot
from pcoa_plots import create_pcoa_plot
import codecs
import sys
import warnings
import os
from pathlib import Path


# Ignore all warnings
warnings.filterwarnings("ignore")

# Ensure standard output uses UTF-8 encoding
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)

os.chdir(sys.argv[1])
path_list = [sys.argv[2], sys.argv[3]]

for path in path_list:
    #### Create results directory if doesn't yet exist ###
    Path(f"{path}/results_unprocessed_asv").mkdir(parents=True, exist_ok=True)

    ### Process tables ###
    df_list = process_data.process_tables(path)
    # df_list = process_data.process_tables_qiime(path)

    ### Create plots ###
    # create_venn_diagram(path)
    # sys.stdout.flush()  # So print statements show up while code is running
    # create_taxa_barplot(df_list, path)
    # sys.stdout.flush()
    create_network_plots(path, df_list, with_labels=True)
    sys.stdout.flush()
    # create_taxa_boxplot(path, df_list)
    # sys.stdout.flush()
    # create_alpha_diversity_boxplot(path)
    # sys.stdout.flush()
    # create_pcoa_plot(path, df_list)

    ## Create plots for Bacteria + Archaea analyses ##
    combined_analysis.create_Ba_Ar_networks(path_list[0], path_list[1], with_labels=True)


# EXAMPLE COMMAND LINE
# ./driver_script.sh C:/Users/gebeb/Desktop/Research/16S_rRNA/RM_nextflow Nextflow_Ba Nextflow_Ar