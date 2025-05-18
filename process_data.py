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


#########################
#####   FUNCTIONS   #####
#########################

def process_tables(path):
    """
    Assumes ASV table is named 'feature-table.tsv' and 1st row has something like '# Constructed from biom file' that is skipped.
    Assumes taxonomy table is named 'taxonomy.tsv' and 2nd row has '#q2:types	categorical	categorical' that is skipped (1st row kept).
    Assumes metadata table is named 'metadata.tsv' and the sample identifiers are in the first column.

    Combines all 3 tables into 1 condensed table. ASVs summed together based on shared genus/species.

    Returns: [genus_df, species_df]
    """
    #################### READ IN THE DATA ###################

    # Read the ASV table
    asv_df = pd.read_csv(f"{path}/feature-table.tsv", sep='\t', skiprows=1, index_col=0)  # set 1st column as index
    #     display(asv_df)

    # Read the taxa table
    taxa_df = pd.read_csv(f"{path}/taxonomy.tsv", sep='\t', skiprows=[1])
    #     display(taxa_df)

    # Read the metadata table
    metadata_df = pd.read_csv("metadata.tsv", sep='\t')
    metadata_df.rename(columns={metadata_df.columns[0]: "Sample_ID"}, inplace=True)
    #     display(metadata_df)

    #################### REFORMAT TAXA TABLE ###################

    # Define taxonomy levels
    taxonomy_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

    # Extract taxonomies to their own columns using regex
    taxa_df[taxonomy_levels] = taxa_df["Taxon"].str.extract(
        r'd__(?P<Kingdom>[^;]*)'
        r'(?:; p__(?P<Phylum>[^;]*))?'
        r'(?:; c__(?P<Class>[^;]*))?'
        r'(?:; o__(?P<Order>[^;]*))?'
        r'(?:; f__(?P<Family>[^;]*))?'
        r'(?:; g__(?P<Genus>[^;]*))?'
        r'(?:; s__(?P<Species>[^;]*))?'
    )

    # Function to replace "uncultured" with the first valid taxonomic rank to the left
    def resolve_uncultured(row):
        cols = ["Species", "Genus", "Family", "Order", "Class", "Phylum", "Kingdom"]
        for i in range(len(cols)):
            if pd.isna(row[cols[i]]) or "uncultured" in str(row[cols[i]]).lower():
                # Find the first valid name to the left
                for j in range(i + 1, len(cols)):
                    if not pd.isna(row[cols[j]]) and "uncultured" not in str(row[cols[j]]).lower():
                        row[cols[i]] = row[cols[j]]
                        break
        return row

    # Apply the function to each row
    taxa_df = taxa_df.apply(resolve_uncultured, axis=1)

    #     display(taxa_df)

    #################### COMBINE ASV AND TAXA TABLES ###################

    # Transpose ASV table so samples are in the first column
    asv_df_T = asv_df.T
    asv_df_T.index.name = 'Sample_ID'  # Rename index for merging

    # Reset index to bring Sample_ID into a column
    asv_df_T = asv_df_T.reset_index()

    # Merge ASV table with taxonomy on Feature ID
    merged_df = asv_df_T.melt(id_vars=['Sample_ID'], var_name='Feature ID', value_name='Count')

    # --- Genus Table ---
    # Merge taxonomy to get genus information
    merged_genus_df = merged_df.merge(taxa_df[['Feature ID', 'Genus']], on='Feature ID', how='left')

    # Aggregate by Sample_ID and Genus (sum ASV counts per genus)
    final_genus_df = merged_genus_df.groupby(['Sample_ID', 'Genus'])['Count'].sum().reset_index()

    # Pivot table to make Genus names the column headers
    final_genus_df = final_genus_df.pivot(index='Sample_ID', columns='Genus', values='Count').fillna(0)

    # Reset index to bring Sample_ID as a column
    final_genus_df = final_genus_df.reset_index()

    # Merge metadata dataframe with the genus abundance table
    genus_df = final_genus_df.merge(metadata_df, on='Sample_ID', how='left')

    # --- Species Table ---
    # Merge taxonomy to get genus information
    merged_species_df = merged_df.merge(taxa_df[['Feature ID', 'Species']], on='Feature ID', how='left')

    # Aggregate by Sample_ID and Genus (sum ASV counts per genus)
    final_species_df = merged_species_df.groupby(['Sample_ID', 'Species'])['Count'].sum().reset_index()

    # Pivot table to make Genus names the column headers
    final_species_df = final_species_df.pivot(index='Sample_ID', columns='Species', values='Count').fillna(0)

    # Reset index to bring Sample_ID as a column
    final_species_df = final_species_df.reset_index()

    # Merge metadata dataframe with the genus abundance table
    species_df = final_species_df.merge(metadata_df, on='Sample_ID', how='left')

    #     display(genus_df)
    #     display(species_df)

    return [genus_df, species_df]


def process_tables_qiime(path):
    """
    input:
        TSV of taxanomic data. Rows are samples, columns are Sample_ID, genus/species names, and metadata.
        EXPECTING THE NAMES OF THESE TABLES: "qiime_genus.tsv" and "qiime_specie.tsv"

    output:
        Pandas Dataframe. Same genus/species columns are combined together by addition.
    """
    # Save the dataframes into a list -- [qiime_genus, qiime_specie]
    qiime_tsvs = []
    df1 = pd.read_csv(f"{path}/qiime_genus.tsv", sep="\t")
    qiime_tsvs.append(df1)
    df2 = pd.read_csv(f"{path}/qiime_specie.tsv", sep="\t")
    qiime_tsvs.append(df2)

    dfs_list = []
    for df in qiime_tsvs:
        # Rename columns to remove "g__" prefix
        df.columns = [col.replace('g__', '').replace('s__', '').replace(' ', '_') for col in df.columns]
        # Reorder rows of dataframe in "numerical order"
        df = df.sort_values(by='Sample_ID', key=lambda col: col.str.extract(r'(\d+)', expand=False).astype(int))
        # Reset the index (optional)
        df = df.reset_index(drop=True)
        # Combine columns with the same base name
        # Create a dictionary to group columns by their base names
        grouped_columns = {}
        for col in df.columns:
            base_name = col.split('.')[0]  # Extract base name (e.g., 'Blautia' from 'Blautia.1')
            if base_name in grouped_columns:
                grouped_columns[base_name].append(col)
            else:
                grouped_columns[base_name] = [col]

        # Create a new dataframe with summed columns
        summed_df = df.copy()
        for base_name, cols in grouped_columns.items():
            if len(cols) > 1:  # If there are multiple columns with the same base name
                summed_df[base_name] = summed_df[cols].sum(axis=1)  # Sum the values row-wise
                summed_df.drop(columns=cols[1:], inplace=True)  # Drop all but the first column

        dfs_list.append(summed_df)

    return dfs_list


def select_asv_columns(df):
    """
    input:
        ASV table. First column is "Sample_ID" or something similar. Last columns are metadata related.

    output:
        Series of column names that correspond to taxa/abundance
    """
    return df.columns.difference(["Sample_ID", "location", "sex", "age", "sample_type",
                                  "breed", "feeding-system", "temp", "sequencing_method"])


def get_genus(asv_name):
    """Extracts genus from ASV column name (e.g., 'Bacteria_Fibrobacter_succinogenes' -> 'Fibrobacter')"""
    return asv_name.split("_")[1] if "_" in asv_name else asv_name


def combine_same_asv_groups(asv_data):
    """Combine and sum together same taxa groups"""
    # Create a dictionary to map original column names to new names
    column_groups = {}
    for col in asv_data.columns:
        # Extract the base name (e.g., 'Prevotellaceae_UCG' from 'Prevotellaceae_UCG-001')
        base_name = re.sub(r'-\d+$', '', col)  # Removes trailing numbers after '-'

        # Group similar names together
        if base_name in column_groups:
            column_groups[base_name].append(col)
        else:
            column_groups[base_name] = [col]

    # Create a new DataFrame with summed values for grouped columns
    asv_data_combined = pd.DataFrame()

    for new_col, old_cols in column_groups.items():
        asv_data_combined[new_col] = asv_data[old_cols].sum(axis=1)

    return asv_data_combined

