#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import jaccard_score, pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import networkx as nx
import networkx.algorithms.community as nx_comm
import markov_clustering as mcl
from rapidfuzz import process, fuzz
import re
import sys
import codecs


####################################################################################
########################## PRE-PROCESSING FUNCTIONS ##############################
####################################################################################

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
    asv_df = pd.read_csv(f"{path}/feature-table.tsv", sep='\t', skiprows=1, index_col=0) # set 1st column as index
#     display(asv_df)
    
    # Read the taxa table
    taxa_df = pd.read_csv(f"{path}/taxonomy.tsv", sep='\t', skiprows=[1])
#     display(taxa_df)
    
    # Read the metadata table
    metadata_df = pd.read_csv("metadata.tsv", sep='\t')
    metadata_df.rename(columns={metadata_df.columns[0]: "SampleID"}, inplace=True)

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
    asv_df_T.index.name = 'SampleID'  # Rename index for merging

    # Reset index to bring SampleID into a column
    asv_df_T = asv_df_T.reset_index()

    # Merge ASV table with taxonomy on Feature ID
    merged_df = asv_df_T.melt(id_vars=['SampleID'], var_name='Feature ID', value_name='Count')

    # --- Genus Table ---
    # Merge taxonomy to get genus information
    merged_genus_df = merged_df.merge(taxa_df[['Feature ID', 'Genus']], on='Feature ID', how='left')

    # Aggregate by SampleID and Genus (sum ASV counts per genus)
    final_genus_df = merged_genus_df.groupby(['SampleID', 'Genus'])['Count'].sum().reset_index()

    # Pivot table to make Genus names the column headers
    final_genus_df = final_genus_df.pivot(index='SampleID', columns='Genus', values='Count').fillna(0)

    # Reset index to bring SampleID as a column
    final_genus_df = final_genus_df.reset_index()
    
    # Merge metadata dataframe with the genus abundance table
    genus_df = final_genus_df.merge(metadata_df, on='SampleID', how='left')

    
    # --- Species Table ---
    # Merge taxonomy to get genus information
    merged_species_df = merged_df.merge(taxa_df[['Feature ID', 'Species']], on='Feature ID', how='left')

    # Aggregate by SampleID and Genus (sum ASV counts per genus)
    final_species_df = merged_species_df.groupby(['SampleID', 'Species'])['Count'].sum().reset_index()

    # Pivot table to make Genus names the column headers
    final_species_df = final_species_df.pivot(index='SampleID', columns='Species', values='Count').fillna(0)

    # Reset index to bring SampleID as a column
    final_species_df = final_species_df.reset_index()
    
    # Merge metadata dataframe with the genus abundance table
    species_df = final_species_df.merge(metadata_df, on='SampleID', how='left')
    
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
    return df.columns.difference(["SampleID", "location", "sex", "age", "sample type", 
                                  "breed", "feeding-system", "temp", "sequencing method"])


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

####################################################################################
########################## SINGLE NETWORK PLOTTING FUNCTION ##############################
####################################################################################

def compute_jaccard_similarity_matrix(graphs):
    """
    Compute the Jaccard similarity matrix for multiple NetworkX graphs.

    Parameters:
        graphs (list): List of networkx.Graph objects.

    Returns:
        pd.DataFrame: DataFrame containing pairwise Jaccard similarity scores.
    """
    num_graphs = len(graphs)
    adj_matrices = [nx.to_numpy_array(G) for G in graphs]

    # Convert adjacency matrices to binary (1 for connected, 0 for not connected)
    adj_matrices = [(adj > 0).astype(int) for adj in adj_matrices]

    # Find max size for padding
    max_size = max(adj.shape[0] for adj in adj_matrices)

    # Pad adjacency matrices to the same size
    adj_matrices = [np.pad(adj, ((0, max_size - adj.shape[0]), (0, max_size - adj.shape[1])), mode='constant')
                    for adj in adj_matrices]

    # Flatten adjacency matrices
    vectors = [adj.flatten() for adj in adj_matrices]

    # Compute Jaccard similarities
    similarity_matrix = np.zeros((num_graphs, num_graphs))
    for i in range(num_graphs):
        for j in range(i, num_graphs):  # Compute only upper triangle (matrix is symmetric)
            similarity = jaccard_score(vectors[i], vectors[j], average='binary')
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Copy to lower triangle

    # Convert to DataFrame for better visualization
    graph_labels = [f"Graph {i+1}" for i in range(num_graphs)]
    similarity_df = pd.DataFrame(similarity_matrix, index=graph_labels, columns=graph_labels)
    
    return similarity_df



def create_network_plots(path, df_list):
    """
    Creates 1 plot with the following network subplots:
        1. Everything
        2. Location - Arusha
        3. Location - Elsewhere
    """
    print("Creating networks...")
    
    i = 0
    metadata_column = 'location'
    metadata_column_values = ['arusha', 'else where']

    # Iterate through [genus_df, species_df]
    for df in df_list:
        if i%2 == 0:
            print("--- GENUS ---")
            sys.stdout.flush()
            G_list = []
        else:
            print("--- SPECIES ---")
            sys.stdout.flush()
            G_list = []
            
        # Initialize a dataframe with topological features
        # At each iteration of the below for loop, add new columns
        topological_feat_df = pd.DataFrame({
            "Topological features": ["Nodes", "Edges", "Positives", "Negatives", "Modularity", 
                                     "Network diameter", "Average degree", "Clustering coefficient"]
        })
        
        # Create the 3 plots: arusha, else where
        for col_val_i in range(len(metadata_column_values)+1):
            df_original = df.copy()
            if col_val_i != 2:
                metadata_value = metadata_column_values[col_val_i]
                print(f"Working with: {metadata_value}")
                sys.stdout.flush()

                # Ensure filtering works correctly
                filtered_df = df_original[df_original[metadata_column].str.strip().str.lower() == metadata_value.lower()]
            else:
                print("Working with: Everything")
                metadata_value = "Everything"
                filtered_df = df_original  # Use the full dataframe
            
#             display(filtered_df)

            # Extract ASV data
            asv_data_columns = select_asv_columns(filtered_df)
            asv_data = filtered_df[asv_data_columns].fillna(0)
            
            # Combine and sum together same groups
            asv_data_combined = combine_same_asv_groups(asv_data)

            # Compute Spearman correlation
            cor_matrix = asv_data_combined.corr(method='spearman')
            
            # Save the dataframe to the results folder
            Path(f"{path}/results").mkdir(parents=True, exist_ok=True)
            
            # Save the correlation matrix
            if i%2 == 0: # name it with 'Genus'
                cor_matrix.to_csv(f"{path}/results/network_correlation_table_Nextflow_Genus_{metadata_value}.csv", index=True)
            else: # name it with 'Species'
                cor_matrix.to_csv(f"{path}/results/network_correlation_table_Nextflow_Species_{metadata_value}.csv", index=True)
            
            # Thresholding
            threshold = 0.3
            edges = cor_matrix[(cor_matrix > threshold) & (cor_matrix != 1)].stack().reset_index()
            edges.columns = ["ASV1", "ASV2", "weight"]

            # Create Graph
            G = nx.Graph()
            for _, row in edges.iterrows():
                G.add_edge(row["ASV1"], row["ASV2"], weight=row["weight"])

            # -- Obtain network data --
            data_degree = nx.degree_centrality(G)
            data_eigenvector = nx.eigenvector_centrality(G)
            data_closeness = nx.closeness_centrality(G)
            data_betweenness = nx.betweenness_centrality(G)

            # Convert dictionaries to DataFrames
            df_degree = pd.DataFrame.from_dict(data_degree, orient='index', columns=['Degree Centrality'])
            df_eigenvector = pd.DataFrame.from_dict(data_eigenvector, orient='index', columns=['Eigenvector Centrality'])
            df_closeness = pd.DataFrame.from_dict(data_closeness, orient='index', columns=['Closeness Centrality'])
            df_betweenness = pd.DataFrame.from_dict(data_betweenness, orient='index', columns=['Betweenness Centrality'])

            # Combine all dataframes into one
            df_combined = df_degree.join([df_eigenvector, df_closeness, df_betweenness])

            # Reset index to turn 'Species' into a column
            df_combined.reset_index(inplace=True)
            df_combined.rename(columns={'index': 'Species'}, inplace=True)
            
            if i%2 == 0: # name it with 'Genus'
                df_combined.to_csv(f"{path}/results/network_centrality_stats_Nextflow_Genus_{metadata_value}.csv", index=False)
            else: # name it with 'Species'
                df_combined.to_csv(f"{path}/results/network_centrality_stats_Nextflow_Species_{metadata_value}.csv", index=False)

            # -- Obtain global properties of the network --
            data_avg_path_length = nx.average_shortest_path_length(G)
            data_clustering_coeff = round(nx.average_clustering(G), 2)
            
            # Detect communities using the Louvain method
            partition = nx.community.louvain_communities(G, seed=123)
            # Compute modularity
            data_modularity = round(nx.algorithms.community.modularity(G, partition), 4)
            
            data_density = nx.density(G)
            data_node_connectivity = nx.node_connectivity(G)
            data_edge_connectivity = nx.edge_connectivity(G)

            global_data = {"Global Properties": [data_avg_path_length, data_clustering_coeff, data_modularity,
                          data_density, data_node_connectivity, data_edge_connectivity]}
            global_data_df = pd.DataFrame(global_data, index=["Average Path Length", "Clustering Coefficient", "Modularity",
                                                              "Density", "Node Connectivity", "Edge Connectivity"])
#             display(global_data_df)
            if i%2 == 0: # name it with 'Genus'
                global_data_df.to_csv(f"{path}/results/network_GlobalProperties_Nextflow_Genus_{metadata_value}.csv")   
            else: # name it with 'Species'
                global_data_df.to_csv(f"{path}/results/network_GlobalProperties_Nextflow_Species_{metadata_value}.csv")  

                
            # ----- Obtain topological features of the network -----
            data_num_nodes = G.number_of_nodes()
            data_num_edges = G.number_of_edges()
            
            ## Get positive and negative correlations
            # Extract the upper triangle of the correlation matrix (excluding the diagonal)
            upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

            # Count positive and negative correlations
            data_num_positive = (upper_triangle > 0).sum().sum()
            data_num_negative = (upper_triangle < 0).sum().sum()

            # Calculate percentages
            total_comparisons = data_num_positive + data_num_negative  # Total valid correlations
            data_num_positive_perc = round((data_num_positive / total_comparisons) * 100 if total_comparisons > 0 else 0, 2)
            data_num_negative_perc = round((data_num_negative / total_comparisons) * 100 if total_comparisons > 0 else 0, 2)
            
            # Join number of positives and percentage as str
            data_positives_str = f"{data_num_positive} ({data_num_positive_perc}%)"
            data_negatives_str = f"{data_num_negative} ({data_num_negative_perc}%)"
            
            # Modularity already calculated in variable "data_modularity"
            
            # Calculate network diameter
            data_diameter = nx.diameter(G)
            
            # Calculate average degree
            data_avg_degree = round(np.mean(list(nx.average_degree_connectivity(G).values())), 2)
            
            # Calculate weighted degree
            # ???? What is this ????
            
            # Clustering coefficient already calculated in variable "data_clustering_coeff"
            
            # Condense topological data to list
            topolog_data_list = [data_num_nodes, data_num_edges, data_positives_str, data_negatives_str, data_modularity, data_diameter, data_avg_degree, data_clustering_coeff]
            
            # Append column to topological dataframe:
            topological_feat_df[metadata_value] = topolog_data_list
            
            # Convert to adjacency matrix for MCL
            adjacency_matrix = nx.to_scipy_sparse_array(G, dtype=float)
            result = mcl.run_mcl(adjacency_matrix, inflation=1.8)
            clusters = mcl.get_clusters(result)

            # Assign clusters
            cluster_map = {}
            node_list = list(G.nodes())
            for c, cluster in enumerate(clusters):
                for node in cluster:
                    cluster_map[node_list[node]] = c 

            # Assign colors to clusters
            num_clusters = len(clusters)
            color_palette = plt.get_cmap("tab20c", num_clusters)
            node_colors_mcl = [color_palette(cluster_map[node]) for node in G.nodes]

            # Compute with all NetworkX Layouts
            layout_type_list = ["spring", "shell", "circular"]
            for layout_type in layout_type_list:
                if layout_type == "spring":
#                     print("Spring Layout:")
                    pos = nx.spring_layout(G, seed=42)
                elif layout_type == "shell":
#                     print("Shell Layout:")
                    pos = nx.shell_layout(G)
                elif layout_type == "circular":
#                     print("Circular Layout:")
                    pos = nx.circular_layout(G)

                node_sizes = [nx.degree_centrality(G)[node] * 500 for node in G.nodes]  

                # Edge Color and Width
                edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
                norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
                cmap = mcolors.LinearSegmentedColormap.from_list("corr_cmap", ["red", "white", "green"])
                edge_colors = [cmap(norm(w)) for w in edge_weights]
                edge_widths = [abs(w) * 5 for w in edge_weights]
                
                
                # --- Plot 1: Color by MCL Cluster ---
                plt.figure(figsize=(12, 8))
                nx.draw_networkx_nodes(G, pos, node_color=node_colors_mcl, node_size=node_sizes)
                nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
                
                # Add labels
#                 if (layout_type == "shell") or (layout_type == "circular"):  
#                     theta = {k: np.arctan2(v[1], v[0]) * 180/np.pi for k, v in pos.items() }
#                     labels = nx.draw_networkx_labels(G, pos, font_size=6)
#                     for key,t in labels.items():
#                         if 90 < theta[key] or theta[key] < -90 :
#                             angle = 180 + theta[key]
#                             t.set_ha('right')
#                         else:
#                             angle = theta[key]
#                             t.set_ha('left')
#                         t.set_va('center')
#                         t.set_rotation(angle)
#                         t.set_rotation_mode('anchor')
#                 else:
#                     nx.draw_networkx_labels(G, pos, font_size=6)

                # Legend
                legend_patches_mcl = [mpatches.Patch(color=color_palette(c), label=f"Cluster {c}") for c in range(num_clusters)]
                plt.legend(handles=legend_patches_mcl, title="MCL Clusters", loc="upper right", bbox_to_anchor=(1.15, 1))
                plt.axis('off')  # Hide the axis (black border around network)
                
                # Save the fig
                Path(f"{path}/results_labeled").mkdir(parents=True, exist_ok=True)
                if i%2 == 0: # name it with 'Genus'
                    plt.title(f"Genus ASV Network - MCL Clustering")
                    plt.savefig(f"{path}/results/network_Nextflow_Genus_{metadata_value}_{layout_type}_MCLColor.png", bbox_inches="tight", dpi=300)
                else:
                    plt.title(f"Species ASV Network - MCL Clustering")
                    plt.savefig(f"{path}/results/network_Nextflow_Species_{metadata_value}_{layout_type}_MCLColor.png", bbox_inches="tight", dpi=300)
                    
#                 plt.show()
                plt.close()

                # --- Plot 2: Color by Genus/Species ---
                plt.figure(figsize=(12, 8))
                unique_genera = list(set(get_genus(node) for node in G.nodes))
                cmap = plt.cm.viridis
                color_map_genus = {genus: cmap(g / len(unique_genera)) for g, genus in enumerate(unique_genera)}
                node_colors_genus = [color_map_genus[get_genus(node)] for node in G.nodes]

                # Draw nodes separately to keep them fully opaque
                nx.draw_networkx_nodes(G, pos, node_color=node_colors_genus, node_size=node_sizes)
                # Draw edges with transparency
                nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)  # Adjust alpha as needed
                # Draw labels
#                 if (layout_type == "shell") or (layout_type == "circular"):  
#                     theta = {k: np.arctan2(v[1], v[0]) * 180/np.pi for k, v in pos.items() }
#                     labels = nx.draw_networkx_labels(G, pos, font_size=6)
#                     for key,t in labels.items():
#                         if 90 < theta[key] or theta[key] < -90 :
#                             angle = 180 + theta[key]
#                             t.set_ha('right')
#                         else:
#                             angle = theta[key]
#                             t.set_ha('left')
#                         t.set_va('center')
#                         t.set_rotation(angle)
#                         t.set_rotation_mode('anchor')
#                 else:
#                     nx.draw_networkx_labels(G, pos, font_size=6)

                legend_patches_genus = [mpatches.Patch(color=color_map_genus[genus], label=genus) for genus in unique_genera]
                if i%2 == 0: # legend labeled with "Genus"
                    plt.title(f"Genus ASV Network")
                    plt.legend(handles=legend_patches_genus, title="Genus", loc="upper right", bbox_to_anchor=(1.4, 1))
                else: # Legend labeled with "Species"
                    plt.title(f"Species ASV Network")
                    plt.legend(handles=legend_patches_genus, title="Species", loc="upper right", bbox_to_anchor=(1.4, 1))

                plt.axis('off')  # Hide the axis (black border around network)
                
                if i%2 == 0: # name it with 'Genus'
                    plt.savefig(f"{path}/results/network_Nextflow_Genus_{metadata_value}_{layout_type}_GenusColor.png", bbox_inches="tight", dpi=300)
                else:
                    plt.savefig(f"{path}/results/network_Nextflow_Species_{metadata_value}_{layout_type}_SpeciesColor.png", bbox_inches="tight", dpi=300)
                
#                 plt.show()
                plt.close()
                
                G_list.append(G)
                
        # display(topological_feat_df)
        # Save the topological dataframe
        if i%2 == 0: # name it with 'Genus'
            topological_feat_df.to_csv(f"{path}/results/network_Topology_Nextflow_Genus.csv", index=False)
            # Compute Jaccard Index with all genus-level networkx graphs
            similarity_df = compute_jaccard_similarity_matrix(G_list)
            similarity_df.to_csv(f"{path}/results/network_JaccardIndex_Nextflow_Genus.csv", index=True)
        else: # name it with 'Species'
            topological_feat_df.to_csv(f"{path}/results/network_Topology_Nextflow_Species.csv", index=False)
            similarity_df = compute_jaccard_similarity_matrix(G_list)
            similarity_df.to_csv(f"{path}/results/network_JaccardIndex_Nextflow_Species.csv", index=True)
            
        i += 1

    print("")
    print("Finished generating network plots and their data tables (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    print("")
    sys.stdout.flush()

    
####################################################################################
########################## TAXA BOXLOT ##############################
####################################################################################

def create_taxa_boxplot(path, df_list):
    """
    input:
        Pandas Dataframe with condensed and added genus/species columns
    output:
        Boxplot. Y-axis is relative abundance of genus/species .
        X-axis are the different genus/species names.
        For each genus/species name is a boxplot for each unique value under the specified metadata column/label
    """
    print("Creating taxa boxplots...")
    sys.stdout.flush()
    
    i = 0
    metadata_column = 'location'
    metadata_column_values = ['arusha', 'else where']

    # Iterate through [genus_df, species_df]
    for df in df_list:
        if i%2 == 0:
            print("--- GENUS ---")
            taxa_lvl = "Genus"
        else:
            print("--- SPECIES ---")
            taxa_lvl = "Species"
                
        # Select numeric columns for relative abundance calculation
        taxa_columns = select_asv_columns(df)  # Between 'Sample_ID' and 'location'

        # Calculate relative abundance
        df[taxa_columns] = df[taxa_columns].div(df[taxa_columns].sum(axis=1), axis=0)

        # Ensure 'SampleID' or 'Sample_ID' is selected
        metadata_columns = [col for col in ['SampleID', 'Sample_ID'] if col in df.columns]
        metadata_columns.append(metadata_column.lower())  # Add the required metadata column

        # Melt the dataframe into a long format
        long_df = pd.melt(
            df,
            id_vars=metadata_columns,  # Metadata to keep
            value_vars=taxa_columns,  # Genus columns to reshape
            var_name=taxa_lvl,          # New column for genus names
            value_name='Relative_Abundance'  # New column for numeric values
        )

        # Plot the boxplots
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=long_df,
            x=taxa_lvl,
            y='Relative_Abundance',
            hue=metadata_column.lower(),  # Create a separate boxplot for each breed
            dodge=True
        )

        # Customize the plot
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.xlabel(taxa_lvl, fontsize=14)
        plt.ylabel('Relative Abundance', fontsize=14)
        plt.legend(title=metadata_column)
        plt.tight_layout()
        plt.savefig(f"{path}/results/taxa_boxplot_Nextflow_QIIME_{taxa_lvl}.png", bbox_inches="tight", dpi=300)
        plt.close()  # Close figure to prevent overlapping plots

        
        # --- Boxplots for functionally related taxa groups ---
        
        # Function to get best matches for taxa
        def get_best_matches(taxa_list, available_taxa, threshold=40):
            best_matches = {}
            for taxon in taxa_list:
                match, score, _ = process.extractOne(taxon, available_taxa, scorer=fuzz.token_sort_ratio)
                if score >= threshold:  # Only take matches with a high enough confidence
                    best_matches[taxon] = match
                else:
                    print(f"⚠️ No good match found for: {taxon} (best score: {score})")
                    sys.stdout.flush()
            return best_matches
            
        if i%2 == 1: # If at species level
            # List of taxa to plot
            hydrogen_producers_taxa = [
                "Fibrobacter succinogenes",
                "Bacteroides ruminicola",
                "Clostridium tetani",
                "Ruminococcus albus"
            ]

            hydrogen_utilizers_taxa = [
                "Selenomonas ruminantium",
                "Succiniclasticum ruminis"
            ]

            taxa_dict = {
                "Hydrogen Producers": hydrogen_producers_taxa,
                "Hydrogen Utilizers": hydrogen_utilizers_taxa
            }

            # Get column names from df
            available_taxa = df.columns.tolist()

            for taxa_group_key, selected_taxa in taxa_dict.items():
                df_copy = df.copy()
                # Get best matching column names
                best_matches = get_best_matches(selected_taxa, available_taxa)

                # Extract matched columns
                matched_columns = list(best_matches.values())

                # Ensure metadata columns are included
                metadata_columns = ['SampleID', 'location', 'sex', 'age', 'sample type', 'breed', 'feeding-system', 'temp', 'sequencing method']
                
                # Subset df with matched columns + metadata
                df_filtered = df_copy[metadata_columns + matched_columns]
                                
                # Convert data to long format
                long_df_taxa_group = pd.melt(
                    df_filtered,
                    id_vars=metadata_columns,
                    value_vars=matched_columns,
                    var_name="Taxa", 
                    value_name="Relative_Abundance"
                )

                # Compute p-value for group differences
                hue_groups = subset[metadata_column.lower()].unique()
                group_values = [subset[subset[metadata_column.lower()] == group]["Relative_Abundance"] for group in hue_groups]

                if len(group_values) == 2:  # Two groups → t-test
                    p_val = ttest_ind(group_values[0], group_values[1], equal_var=False).pvalue
                else:
                    p_val = None  # Extend to ANOVA/Kruskal-Wallis if more than two groups

                # Plot boxplot
                plt.figure(figsize=(8, 6))
                ax = sns.boxplot(
                    data=subset,
                    x=metadata_column.lower(),
                    y='Relative_Abundance',
                    dodge=True
                )

                # Annotate p-value if available
                if p_val is not None:
                    ax.text(0.5, max(subset["Relative_Abundance"]) * 0.9, f"p={p_val:.3f}",
                            horizontalalignment='center', color='black', fontsize=12)

                # Customizations
                plt.xticks(rotation=45, ha='right')
                plt.xlabel(metadata_column, fontsize=14)
                plt.ylabel('Relative Abundance', fontsize=14)
                plt.title(f"{taxa_group_key}")
                plt.tight_layout()

                # Save the figure
                plt.savefig(f"{path}/results/boxplot_Nextflow_QIIME_{taxa_group_key}.png", 
                            bbox_inches="tight", dpi=300)
                plt.close()  # Close figure to prevent overlapping plots

                
        # ----- Boxplots for selected taxa -----
        
        # List of specific taxa to plot
        selected_taxa = [
            "Fibrobacter succinogenes",
            "Bacteroides ruminicola",
            "Ruminococcus albus",
            "Eubacterium ruminantium",
            "Selenomonas ruminantium",
            "Succiniclasticum ruminis",
            "Methanobrevibacter ruminantium",
            "Methanobrevibacter gottschalkii",
            "Methanomicrobium mobile",
            "Methanosphaera cuniculi"
        ]

        # Ensure taxa names are clean
        long_df[taxa_lvl] = long_df[taxa_lvl].str.strip().str.lower()
        available_taxa = long_df[taxa_lvl].unique()

        # Get best matches
        taxa_mapping = get_best_matches([taxon.lower() for taxon in selected_taxa], available_taxa)

        # Replace selected taxa with their best matches
        matched_taxa = list(taxa_mapping.values())

#         Print mapping for verification
        print("🔍 Taxa Mapping (Original → Matched):")
        for original, matched in taxa_mapping.items():
            print(f"{original} → {matched}")
            sys.stdout.flush()

        # Loop through each matched taxon to create separate plots
        for taxon in matched_taxa:
            subset = long_df[long_df[taxa_lvl] == taxon]  # Filter data for this taxon

            if subset.empty:
                print(f"⚠️ Skipping {taxon}: No data available.")
                sys.stdout.flush()
                continue

            # Compute p-value for group differences
            hue_groups = subset[metadata_column.lower()].unique()
            group_values = [subset[subset[metadata_column.lower()] == group]["Relative_Abundance"] for group in hue_groups]

            if len(group_values) == 2:  # Two groups → t-test
                p_val = ttest_ind(group_values[0], group_values[1], equal_var=False).pvalue
            else:
                p_val = None  # Extend to ANOVA/Kruskal-Wallis if more than two groups

            # Plot boxplot
            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(
                data=subset,
                x=metadata_column.lower(),
                y='Relative_Abundance',
                dodge=True
            )

            # Annotate p-value if available
            if p_val is not None:
                ax.text(0.5, max(subset["Relative_Abundance"]) * 0.9, f"p={p_val:.3f}",
                        horizontalalignment='center', color='black', fontsize=12)

            # Customizations
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(metadata_column, fontsize=14)
            plt.ylabel('Relative Abundance', fontsize=14)
            plt.title(f"{taxon} - {taxa_lvl} Level")
            plt.tight_layout()

            # Save the figure
            plt.savefig(f"{path}/results/boxplot_{taxon.replace(' ', '_')}_Nextflow_QIIME_{taxa_lvl}.png", 
                        bbox_inches="tight", dpi=300)
            plt.close()  # Close figure to prevent overlapping plots
            
        
        i += 1
    
    print("")
    print("Finished generating boxplots ٩(◕‿◕｡)۶")
    print("")
    sys.stdout.flush()

    
####################################################################################
########################## ALPHA DIVERSITY BOXPLOTS ##############################
####################################################################################    

def create_alpha_diversity_boxplot(path):
    print("")
    print("Generating alpha diversity boxplots...")
    sys.stdout.flush()
    
    # --- Preprocessing alpha diversity tables ---
    alpha_dfs = []
    chao1_df = pd.read_csv(f"{path}/alpha_diversity_outputs/chao1_alpha_diversity.tsv", sep="\t")
    observed_asvs_df = pd.read_csv(f"{path}/alpha_diversity_outputs/observed_ASVs.tsv", sep="\t")
    shannon_df = pd.read_csv(f"{path}/alpha_diversity_outputs/shannon_vector_alpha_diversity.tsv", sep="\t")
    simpson_df = pd.read_csv(f"{path}/alpha_diversity_outputs/simpson_alpha_diversity.tsv", sep="\t")
    alpha_dfs.extend([chao1_df, observed_asvs_df, shannon_df, simpson_df])
    
    # Read in metadata
    alpha_metadata_df = pd.read_csv("metadata.tsv", sep="\t")
    
    # Iterate over each alpha diversity dataframe
    for df in alpha_dfs:
        # Extract only the first two columns (Sample ID and the alpha diversity metric)
        df = df.iloc[:, :2]  
        # Merge on the sample ID column
        alpha_metadata_df = alpha_metadata_df.merge(df, left_on='sample-id', right_on=df.columns[0], how='left')
        # Drop duplicate Sample ID column from the alpha dataframe
        alpha_metadata_df.drop(columns=[df.columns[0]], inplace=True)

        
    # Function to extract the mean from the tuple-like values in 'chao1_ci'
    def extract_chao1_mean(value):
        if isinstance(value, str):  # Ensure it's a string before processing
            value = value.strip("()").split(",")  # Remove parentheses and split values
            return np.mean([float(value[0]), float(value[1])])  # Compute the mean
        return np.nan  # Handle unexpected cases

    # Apply the function to the 'chao1_ci' column
    alpha_metadata_df["chao1_ci"] = alpha_metadata_df["chao1_ci"].apply(extract_chao1_mean)
    
    
    # --- Create separate boxplots for arusha and else where ---
    # Define metrics and location column
    metrics = ["chao1_ci", "observed_features", "shannon_entropy", "simpson"]
    location_col = "location"

    # Convert 'location' column to categorical with Arusha vs Elsewhere
    alpha_metadata_df[location_col] = alpha_metadata_df[location_col].astype(str)
    alpha_metadata_df[location_col] = alpha_metadata_df[location_col].replace({'arusha': 'Arusha', 'else where': 'Elsewhere'})

    # Create boxplots
    plt.figure(figsize=(12, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)  # Create a 2x2 subplot grid

        # Perform Mann-Whitney U test
        group_1 = alpha_metadata_df[alpha_metadata_df[location_col] == "Arusha"][metric].dropna()
        group_2 = alpha_metadata_df[alpha_metadata_df[location_col] == "Elsewhere"][metric].dropna()

        if len(group_1) > 0 and len(group_2) > 0:  # Ensure there are values to compare
            stat, p_val = mannwhitneyu(group_1, group_2, alternative="two-sided")
            p_text = f"p = {p_val:.3f}"
        else:
            p_text = "p = N/A"

        # Create boxplot
        ax = sns.boxplot(data=alpha_metadata_df, x=location_col, y=metric, palette="plasma")

        # Get y-axis limits and adjust p-value position
        y_min, y_max = ax.get_ylim()
        y_text = y_max - ((y_max - y_min) * 0.1)  # Keep text within 90% of the plot

        # Annotate p-value inside the black border
        ax.text(0.5, y_text, p_text, horizontalalignment='center', fontsize=12, color='black') # bold font: fontweight='bold'

        # Labels and title
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xlabel("Location")
        plt.ylabel(metric.replace('_', ' ').title())

    plt.tight_layout()
    plt.savefig(f"{path}/results/alpha_diversity_boxplot.png", dpi=300, bbox_inches="tight")
    
    print("Finsihed generating alpha diversity box plots ദ്ദി(｡•̀ ,<)~✩‧₊")
    sys.stdout.flush()

    
####################################################################################
########################## FUCNTIONS FOR COMBINED ANALYSIS ##############################
####################################################################################

def combine_datasets(df1, df2):
    """
    Given 2 datasets with the same "Sample_ID" column, in the same order, and with the same metadata columns, combine the 2 datasets.
    Removes "s__" or "g__" prefixes, and adds prefixes to specify bacteria and archaea
    
    Input:
        df1: Pandas dataframe of BACTERIAL data
        df2: Pandas dataframe of ARCHAEA data
    
    Output:
        A single Pandas dataframe
    """
#     # Clean up and add like genus/species columns together
#     df1_summed = modify_table(df1)
#     df2_summed = modify_table(df2)

    # Ensure Sample_ID exists
    if "SampleID" not in df1.columns or "SampleID" not in df2.columns:
        raise ValueError("SampleID column is missing from one of the datasets!")

    # Label the columns of each dataframe with "Bacteria_" or "Archaea_"
    columns_to_modify_ba = select_asv_columns(df1)  # Select ASV columns
    new_column_names_ba = {col: f"Bacteria_{col}" for col in columns_to_modify_ba}
    df1.rename(columns=new_column_names_ba, inplace=True)
    
    columns_to_modify_ar = select_asv_columns(df2)
    new_column_names_ar = {col: f"Archaea_{col}" for col in columns_to_modify_ar}
    df2.rename(columns=new_column_names_ar, inplace=True)
    
    # Identify metadata columns dynamically, ensuring ASV columns are excluded
    all_columns = set(df1.columns)
    non_asv_columns = all_columns - set(list(new_column_names_ba.values()))  # Exclude ASV columns
    metadata_cols = list(non_asv_columns - {"SampleID"})  # Ensure Sample_ID is not included

    # Merge on "Sample_ID" while keeping only one copy of the metadata columns
    df_combined = pd.merge(
        df1.drop(columns=metadata_cols, errors="ignore"), 
        df2,  # Drop metadata from df2
        on="SampleID",
        how="inner"  # Ensures only matching samples are retained
    )

    return df_combined
    
    
def create_Ba_Ar_networks(ba_path, ar_path, path='.'):
    
    ############# PRE-PROCESSING #############
    
    ba_df_list = process_tables(ba_path)
    ar_df_list = process_tables(ar_path)
    
    # Combine Ba and Ar tables --> [Ba_Ar_genus_df, Ba_Ar_species_df]
    combined_df_list = []
    for i in range(len(ba_df_list)):
        combined_df = combine_datasets(ba_df_list[i], ar_df_list[i])
        combined_df_list.append(combined_df)
    
    ############# GENERATE NETWORKS #############
    
    create_network_plots(path, combined_df_list)
    
    return None


def create_Ba_Ar_boxplots(ba_path, ar_path, path='.'):
    
    ############# PRE-PROCESSING #############
    
    ba_df_list = process_tables(ba_path)
    ar_df_list = process_tables(ar_path)
    
    # Combine Ba and Ar tables --> [Ba_Ar_genus_df, Ba_Ar_species_df]
    combined_df_list = []
    for i in range(len(ba_df_list)):
        combined_df = combine_datasets(ba_df_list[i], ar_df_list[i])
        combined_df_list.append(combined_df)
    
    ############# GENERATE NETWORKS #############
    
    create_taxa_boxplot(path, combined_df_list)
    
    return None


####################################################################################
########################## BETA DIVERSITY PLOTS ##############################
####################################################################################

def draw_ellipse(x, y, ax, n_std=1.96, **kwargs):
    if len(x) > 1:  # Ellipse needs at least 2 points
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                          width=lambda_[0]*n_std*2, 
                          height=lambda_[1]*n_std*2,
                          angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                          alpha=0.25,
                          **kwargs)
        ax.add_patch(ellipse)



def create_pcoa_plot(path, df_list, palette="Set2"):
    """ Generates a Bray-Curtis PCoA plot with confidence ellipses per location. """
    i = 0
    for df in df_list:   # [genus_df, species_df]
        # ---- Genus/Species labeling ---
        if i%2 == 0:
            taxa_lvl = "Genus"
        else:
            taxa_lvl = "Species"
        
        # ---- Extract ASV data ----
        asv_data_columns = select_asv_columns(df)
        asv_data = df[asv_data_columns].fillna(0)

        # ---- Calculate Bray-Curtis Distance Matrix ----
        bc_dist = squareform(pdist(asv_data, metric="braycurtis"))

        # ---- Perform Principal Coordinate Analysis (PCoA) Using PCA ----
        pcoa = PCA(n_components=2)
        pcoa_results = pcoa.fit_transform(bc_dist)
        
        # Get variance explained by each axis
        variance_explained = pcoa.explained_variance_ratio_ * 100  # Convert to percentage
        pc1_var = round(variance_explained[0], 2)
        pc2_var = round(variance_explained[1], 2)
        
        # Convert to DataFrame
        pcoa_df = pd.DataFrame(pcoa_results, columns=["PCoA1", "PCoA2"], index=df.index)
        pcoa_df["location"] = df["location"].values  # Add location info

        # ---- Set Color Palette ----
        unique_locations = pcoa_df["location"].unique()
        palette_colors = sns.color_palette(palette, len(unique_locations))
        color_dict = dict(zip(unique_locations, palette_colors))  # Map locations to colors

        # ---- Plot PCoA with Color Coding ----
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Scatter plot colored by 'location'
        sns.scatterplot(data=pcoa_df, x="PCoA1", y="PCoA2", hue="location", s=100, edgecolor="black", palette=color_dict)

        # ---- Add Confidence Ellipses Per Location ----
        for location, group in pcoa_df.groupby("location"):
            draw_ellipse(group["PCoA1"], group["PCoA2"], ax, color=color_dict[location])

        plt.xlabel(f"PCoA1 ({pc1_var}%)")
        plt.ylabel(f"PCoA2 ({pc2_var}%)")
        plt.title(f"PCoA Plot (Bray-Curtis Distance) - {taxa_lvl} Level")
        plt.legend(title="Location")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/results/pcoa_bray_curtis_{taxa_lvl}.png", dpi=300, bbox_inches="tight")
        
        i += 1
    
    print("Done creating PcoA plots ( ˶ˆᗜˆ˵ )")
    sys.stdout.flush()
    
            
def create_Ba_Ar_pcoa_plot(ba_path, ar_path, path='.'):
    print("------ Processing combined Ba and Ar samples ------")
    
    ############# PRE-PROCESSING #############
    
    ba_df_list = process_tables(ba_path)
    ar_df_list = process_tables(ar_path)
    
    # Combine Ba and Ar tables --> [Ba_Ar_genus_df, Ba_Ar_species_df]
    combined_df_list = []
    for i in range(len(ba_df_list)):
        combined_df = combine_datasets(ba_df_list[i], ar_df_list[i])
        combined_df_list.append(combined_df)
    
    ############# GENERATE PCOA PLOTS #############
    
    create_pcoa_plot(path, combined_df_list)
    
    return None


####################################################################################
################################### MAIN ###########################################
####################################################################################

def main():
    # Exit with warning if not enought arguments provided
    if len(sys.argv) < 3:
        sys.stderr.write("\nERROR: Not enough arguments provided in the shell script." + "\n\n")
        sys.stderr.write("\nUSAGE: 16S_rRNA_plotting_pipeline.py </path/BacteriaFolder> </path/ArchaeaFolder>" + "\n\n")
        sys.stderr.write("Generates various plots from 16S rRNA ASVs and taxonomic classification tables. \n\n")
        sys.exit(1)
    
    np.random.seed(816)
    
    # Ensure standard output uses UTF-8 encoding
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)
    
    path_list = [sys.argv[1], sys.argv[2]]

    for path in path_list:
        print(f"------ Processing {path} ------")
        sys.stdout.flush()  # So print statements show up while code is running
        df_list = process_tables(path)  # Use this for original ASV/taxa class table
    #     df_list = process_tables_qiime(path)  # Use this for QIIME2 processed table

        create_network_plots(path, df_list)
        create_taxa_boxplot(path, df_list)
        create_alpha_diversity_boxplot(path)
        create_pcoa_plot(path, df_list)

    # --- For combined bacteria and archaea plots ---
    create_Ba_Ar_networks(path_list[0], path_list[1])
    create_Ba_Ar_boxplots(path_list[0], path_list[1])
    create_Ba_Ar_pcoa_plot(path_list[0], path_list[1])
    
    print("Everything completed successfully!")

if __name__ == "__main__":
    main()
