from process_data import select_asv_columns, combine_same_asv_groups, get_genus
from taxa_barplot_heatmap import create_correlation_heatmap
import os
from pathlib import Path
import pandas as pd
import numpy as np
import community.community_louvain as community
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
import networkx as nx


def create_network_plots(path, df_list, with_labels=False):
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
        if i % 2 == 0:
            taxa_level = "Genus"
            print("--- GENUS ---")
        else:
            taxa_level = "Species"
            print("--- SPECIES ---")

        # Initialize a dataframe with topological features
        # At each iteration of the below for loop, add new columns
        topological_feat_df = pd.DataFrame({
            "Topological features": ["Nodes", "Edges", "Positives", "Negatives", "Modularity",
                                     "Network diameter", "Average degree", "Clustering coefficient"]
        })

        # Create the 3 plots: arusha, else where
        for col_val_i in range(len(metadata_column_values) + 1):
            df_original = df.copy()
            if col_val_i != 2:
                metadata_value = metadata_column_values[col_val_i]
                safe_metadata_value = metadata_value.replace(" ", "_")
                print(f"Working with: {metadata_value}")

                # Ensure filtering works correctly
                filtered_df = df_original[
                    df_original[metadata_column].str.strip().str.lower() == metadata_value.lower()]
            else:
                print("Working with: Everything")
                metadata_value = "Everything"
                safe_metadata_value = "Everything"
                filtered_df = df_original  # Use the full dataframe

            #             display(filtered_df)

            # Extract ASV data
            asv_data_columns = select_asv_columns(filtered_df)
            asv_data = filtered_df[asv_data_columns].fillna(0)

            # Combine and sum together same groups
            asv_data_combined = combine_same_asv_groups(asv_data)

            # Compute Spearman correlation and drop rows/columns with NaN
            cor_matrix = asv_data_combined.corr(method='spearman').dropna(how="all", axis=0).dropna(how="all", axis=1)

            # Save the dataframe to the results folder
            Path(f"{path}/results").mkdir(parents=True, exist_ok=True)

            # Save the correlation matrix
            if i % 2 == 0:  # name it with 'Genus'
                cor_matrix.to_csv(
                    f"{path}/results_data/network_correlation_table_Nextflow_Genus_{safe_metadata_value}.csv", index=True)
            else:  # name it with 'Species'
                cor_matrix.to_csv(
                    f"{path}/results_data/network_correlation_table_Nextflow_Species_{safe_metadata_value}.csv",
                    index=True)

            # Create heatmap from correlation table
            create_correlation_heatmap(cor_matrix, taxa_level, metadata_value, path)

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
            df_eigenvector = pd.DataFrame.from_dict(data_eigenvector, orient='index',
                                                    columns=['Eigenvector Centrality'])
            df_closeness = pd.DataFrame.from_dict(data_closeness, orient='index', columns=['Closeness Centrality'])
            df_betweenness = pd.DataFrame.from_dict(data_betweenness, orient='index',
                                                    columns=['Betweenness Centrality'])

            # Combine all dataframes into one
            df_combined = df_degree.join([df_eigenvector, df_closeness, df_betweenness])

            # Reset index to turn 'Species' into a column
            df_combined.reset_index(inplace=True)
            df_combined.rename(columns={'index': 'Species'}, inplace=True)

            if i % 2 == 0:  # name it with 'Genus'
                df_combined.to_csv(
                    f"{path}/results_data/network_centrality_stats_Nextflow_Genus_{safe_metadata_value}.csv", index=False)
            else:  # name it with 'Species'
                df_combined.to_csv(
                    f"{path}/results_data/network_centrality_stats_Nextflow_Species_{safe_metadata_value}.csv",
                    index=False)

            # -- Obtain global properties of the network --
            # Check if graph is connected
            if not nx.is_connected(G):
                # Get the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                G_lcc = G.subgraph(largest_cc).copy()
                print("Graph not connected. Using largest connected component for average path length.")
            else:
                G_lcc = G

            data_avg_path_length = nx.average_shortest_path_length(G_lcc)
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
            global_data_df = pd.DataFrame(global_data,
                                          index=["Average Path Length", "Clustering Coefficient", "Modularity",
                                                 "Density", "Node Connectivity", "Edge Connectivity"])
            #             display(global_data_df)
            if i % 2 == 0:  # name it with 'Genus'
                global_data_df.to_csv(
                    f"{path}/results_data/network_GlobalProperties_Nextflow_Genus_{safe_metadata_value}.csv")
            else:  # name it with 'Species'
                global_data_df.to_csv(
                    f"{path}/results_data/network_GlobalProperties_Nextflow_Species_{safe_metadata_value}.csv")

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
            data_num_positive_perc = round(
                (data_num_positive / total_comparisons) * 100 if total_comparisons > 0 else 0, 2)
            data_num_negative_perc = round(
                (data_num_negative / total_comparisons) * 100 if total_comparisons > 0 else 0, 2)

            # Join number of positives and percentage as str
            data_positives_str = f"{data_num_positive} ({data_num_positive_perc}%)"
            data_negatives_str = f"{data_num_negative} ({data_num_negative_perc}%)"

            # Modularity already calculated in variable "data_modularity"

            # Calculate network diameter
            data_diameter = nx.diameter(G_lcc)

            # Calculate average degree
            data_avg_degree = round(np.mean(list(nx.average_degree_connectivity(G).values())), 2)

            # Calculate weighted degree
            # ???? What is this ????

            # Clustering coefficient already calculated in variable "data_clustering_coeff"

            # Condense topological data to list
            topolog_data_list = [data_num_nodes, data_num_edges, data_positives_str, data_negatives_str,
                                 data_modularity, data_diameter, data_avg_degree, data_clustering_coeff]

            # Append column to topological dataframe:
            topological_feat_df[metadata_value] = topolog_data_list

            # Perform Louvain clustering
            partition = community.best_partition(G)
            num_clusters = len(set(partition.values()))
            color_palette = plt.get_cmap("tab20c", num_clusters)
            node_colors_louvain = [color_palette(partition[node]) for node in G.nodes]

            # Compute Eigenvector Centrality for Node Size Scaling
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
            scaler = MinMaxScaler(feature_range=(100, 1000))  # Adjust size range
            node_sizes = scaler.fit_transform(np.array(list(eigenvector_centrality.values())).reshape(-1, 1)).flatten()

            edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))

            # Define custom colormap and alpha scaling
            edge_colors = []
            edge_widths = []

            for w in edge_weights:
                if w > 0:  # Positive coxrrelation (Green)
                    edge_colors.append((0, 1, 0, 0.6 + (w * 0.4)))  # 60% to 100% opacity
                    edge_widths.append(1.5 + (w * 2))  # Slightly thicker for strong green edges
                else:  # Negative correlation (Red)
                    edge_colors.append((1, 0, 0, 0.01 + (abs(w) * 0.2)))  # 10% to 30% opacity
                    edge_widths.append(1 + (abs(w) * 1.5))  # Thinner red edges

            # Compute Layouts
            layout_type = 'spring'  # ["spring", "shell", "circular"]
            pos = nx.spring_layout(G, seed=42)
            #             for layout_type in layout_type_list:
            #                 if layout_type == "spring":
            #                     pos = nx.spring_layout(G, seed=42)
            #                 elif layout_type == "shell":
            #                     pos = nx.shell_layout(G)
            #                 elif layout_type == "circular":
            #                     pos = nx.circular_layout(G)

            # Edge Color and Width
            edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            cmap = mcolors.LinearSegmentedColormap.from_list("corr_cmap", ["red", "green"])
            edge_colors = []
            for w in edge_weights:
                base_color = cmap(norm(w))
                # Apply transparency: less opaque for negative (red), more for positive (green)
                if w >= 0:
                    edge_colors.append((*base_color[:3], 0.6 + 0.4 * norm(w)))  # 60–100% for green
                else:
                    edge_colors.append((*base_color[:3], 0.01 + 0.1 * abs(norm(w))))  # 20–40% for red
            edge_widths = [abs(w) * 2 for w in edge_weights]

            # --- Plot: Color by boxplot ---
            fig, ax = plt.subplots(figsize=(12, 8))  # <- Instead of plt.figure()
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_louvain, node_size=node_sizes)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)

            # Add colorbar explicitly to this axis
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, orientation='horizontal', pad=0.15, aspect=40)
            cbar.set_label("Correlation Strength", fontsize=10)

            if with_labels:
                nx.draw_networkx_labels(G, pos, font_size=3)

            # Legend
            legend_patches = [mpatches.Patch(color=color_palette(c), label=f"Cluster {c}") for c in
                              set(partition.values())]
            plt.legend(handles=legend_patches, title="Louvain Clusters", loc="upper right", bbox_to_anchor=(1.15, 1))
            plt.axis('off')
            plt.title(f"{taxa_level} ASV Network - Louvain Clustering")

            # Ensure the directory exists
            output_dir = f"{path}/results"  # f"{path}/results_labeled" if with_labels else f"{path}/results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/network_Nextflow_{taxa_level}_{safe_metadata_value}_{layout_type}_LouvainColor_LABELED.png" if with_labels else f"{output_dir}/network_Nextflow_{taxa_level}_{safe_metadata_value}_{layout_type}_LouvainColor_UNLABELED.png"
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            # --- Plot 2: Color by Genus/Species ---
            fig, ax = plt.subplots(figsize=(12, 8))  # <- Instead of plt.figure()
            unique_genera = list(set(get_genus(node) for node in G.nodes))
            cmap = plt.cm.viridis
            color_map_genus = {genus: cmap(g / len(unique_genera)) for g, genus in enumerate(unique_genera)}
            node_colors_genus = [color_map_genus[get_genus(node)] for node in G.nodes]

            # Draw nodes separately to keep them fully opaque
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_genus, node_size=node_sizes)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)

            if with_labels:
                # Draw labels
                if (layout_type == "shell") or (layout_type == "circular"):
                    theta = {k: np.arctan2(v[1], v[0]) * 180 / np.pi for k, v in pos.items()}
                    labels = nx.draw_networkx_labels(G, pos, font_size=3)
                    for key, t in labels.items():
                        if 90 < theta[key] or theta[key] < -90:
                            angle = 180 + theta[key]
                            t.set_ha('right')
                        else:
                            angle = theta[key]
                            t.set_ha('left')
                        t.set_va('center')
                        t.set_rotation(angle)
                        t.set_rotation_mode('anchor')
                else:
                    nx.draw_networkx_labels(G, pos, font_size=3)

            legend_patches_genus = [mpatches.Patch(color=color_map_genus[genus], label=genus) for genus in
                                    unique_genera]

            plt.title(f"{taxa_level} ASV Network")
            plt.legend(handles=legend_patches_genus, title=taxa_level, loc="upper right", bbox_to_anchor=(1.4, 1))
            plt.axis('off')  # Hide the axis (black border around network)
            output_path = f"{output_dir}/network_Nextflow_{taxa_level}_{safe_metadata_value}_{layout_type}_{taxa_level}Color_LABELED.png" if with_labels else f"{output_dir}/network_Nextflow_{taxa_level}_{safe_metadata_value}_{layout_type}_{taxa_level}Color_UNLABELED.png"
            plt.savefig(output_path, bbox_inches="tight", dpi=300)

            #                 plt.show()
            plt.close()

        #         display(topological_feat_df)
        # Save the topological dataframe
        if with_labels:
            topological_feat_df.to_csv(f"{output_dir}/network_Topology_Nextflow_{taxa_level}_LABELED.csv", index=False)
        else:
            topological_feat_df.to_csv(f"{output_dir}/network_Topology_Nextflow_{taxa_level}_UNLABELED.csv",
                                       index=False)

        i += 1

    print("")
    print("Finished generating network plots and heatmaps (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    print("")