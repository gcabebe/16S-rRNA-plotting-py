from process_data import select_asv_columns
import random
import sys
import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import community.community_louvain as community
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from matplotlib_venn import venn2, venn3
import seaborn as sns
import distinctipy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind, mannwhitneyu, permutation_test
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import networkx.algorithms.community as nx_comm
import markov_clustering as mcl
from rapidfuzz import process, fuzz

def create_taxa_barplot(df_list, path):
    """
    input:
        Pandas Dataframe with condensed and added genus/species columns

    output:
        Relative abundance taxa barplot. Y-axis is relative abundance. X-axis contains each sample.
        Legend shows color codes for each genus/species.
        Saves plot as PNG.
    """
    print("Creating relative abundance taxa barplot...")

    i = 0
    metadata_column = 'location'
    metadata_column_values = ['arusha', 'else where']

    # Iterate through [genus_df, species_df]
    for df in df_list:
        if i % 2 == 0:
            taxa_level = "Genus"
            print("--- GENUS ---")
        else:
            taxa_level = "Species"
            print("--- SPECIES ---")
        # Select numeric columns for relative abundance calculation
        taxa_columns = select_asv_columns(df)  # All columns between 'Sample_ID' and 'location'

        # Calculate relative abundance
        df[taxa_columns] = df[taxa_columns].div(df[taxa_columns].sum(axis=1), axis=0)

        # Melt dataframe for plotting
        melted_df = df.melt(
            id_vars=["Sample_ID"],
            value_vars=taxa_columns,
            var_name=taxa_level,
            value_name="Relative Abundance"
        )

        # Add location info for relabeling
        melted_df["location"] = melted_df["Sample_ID"].map(df.set_index("Sample_ID")["location"])

        # Sum relative abundances per taxon across all samples
        top_taxa = (
            melted_df.groupby(taxa_level)["Relative Abundance"]
                .sum()
                .sort_values(ascending=False)
                .head(30)
                .index
        )

        # Sort samples by location and sample ID
        melted_df["location"] = melted_df["location"].astype(str)
        melted_df = melted_df.sort_values(["location", "Sample_ID"])

        # Plot full barplot
        plt.figure(figsize=(12, 8))
        taxa_palette = dict(zip(top_taxa, distinctipy.get_colors(len(top_taxa))[::-1]))
        ax = sns.barplot(
            data=melted_df[melted_df[taxa_level].isin(top_taxa)],
            x="Sample_ID",
            y="Relative Abundance",
            hue=taxa_level,
            dodge=False,
            palette=taxa_palette
        )

        # Thicken the plot border (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Set x-tick labels based on location
        sample_to_location = df.set_index("Sample_ID")["location"].to_dict()
        xtick_labels = [sample_to_location.get(label.get_text(), "") for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

        # Customize legend (top 20 taxa only)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l in top_taxa]
        if filtered:
            handles, labels = zip(*filtered)
            legend = ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", title=taxa_level)
            legend.get_frame().set_linewidth(2)  # Thicken legend border
            legend.get_frame().set_edgecolor("black")  # Optional: ensure it's visible
            plt.setp(legend.get_title(), fontweight='bold')  # Bold legend title
            for text in legend.get_texts():
                text.set_fontweight('bold')  # Bold legend items
        else:
            ax.legend_.remove()

        # Save to the results folder
        Path(f"{path}/results").mkdir(parents=True, exist_ok=True)

        # Customize plot
        plt.ylabel("Relative Abundance", fontweight='bold')
        plt.xlabel("Samples", fontweight='bold')
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        plt.title("Taxa Relative Abundance Across Samples", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{path}/results/taxa_barplot_{taxa_level}_reordered.png", bbox_inches="tight", dpi=300)
        plt.close()

        i += 1


def create_correlation_heatmap(cor_matrix, taxa_level, metadata_value, path):
    safe_metadata_value = metadata_value.replace(" ", "_")

    #     plt.figure(figsize=(10, 8))
    #     # define the mask to set the values in the upper triangle to True
    #     mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
    #     heatmap = sns.heatmap(cor_matrix, mask=mask, cmap='RdBu')

    #     heatmap.set_title(f'Correlation Heatmap - {metadata_value} on {taxa_level} Level', fontdict={'fontsize':9}, pad=9)
    # #     plt.savefig(f"{path}/results/heatmap_Nextflow_{taxa_level}_{metadata_value}.png", bbox_inches="tight", dpi=300)
    #     plt.show()
    #     plt.close()

    # Save to the results folder
    Path(f"{path}/results").mkdir(parents=True, exist_ok=True)

    clustermap = sns.clustermap(cor_matrix, method="complete", cmap='RdBu', annot_kws={"size": 5}, figsize=(15, 12))
    plt.savefig(f"{path}/results/clustermap_Nextflow_{taxa_level}_{safe_metadata_value}.png", bbox_inches="tight",
                dpi=300)