from process_data import select_asv_columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, permutation_test
from rapidfuzz import process, fuzz

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
    i = 0
    metadata_column = 'location'
    metadata_column_values = ['arusha', 'else where']

    # Iterate through [genus_df, species_df]
    for df in df_list:
        if i % 2 == 0:
            print("--- GENUS ---")
            taxa_lvl = "Genus"
        else:
            print("--- SPECIES ---")
            taxa_lvl = "Species"

        # Select numeric columns for relative abundance calculation
        taxa_columns = select_asv_columns(df)  # Between 'Sample_ID' and 'location'

        # Calculate relative abundance
        df[taxa_columns] = df[taxa_columns].div(df[taxa_columns].sum(axis=1), axis=0)

        # Ensure 'Sample_ID' or 'Sample_ID' is selected
        metadata_columns = [col for col in ['SampleID', 'Sample_ID'] if col in df.columns]
        metadata_columns.append(metadata_column.lower())  # Add the required metadata column

        # Melt the dataframe into a long format
        long_df = pd.melt(
            df,
            id_vars=metadata_columns,  # Metadata to keep
            value_vars=taxa_columns,  # Genus columns to reshape
            var_name=taxa_lvl,  # New column for genus names
            value_name='Relative_Abundance'  # New column for numeric values
        )

        # Plot the boxplots
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(
            data=long_df,
            x=taxa_lvl,
            y='Relative_Abundance',
            hue=metadata_column.lower(),  # Create a separate boxplot for each breed
            dodge=True
        )

        # Thicken the plot border (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Customize legend
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels)]
        legend = ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", title=metadata_column)
        legend.get_frame().set_linewidth(2)  # Thicken legend border
        legend.get_frame().set_edgecolor("black")  # Optional: ensure it's visible
        plt.setp(legend.get_title(), fontweight='bold')  # Bold legend title
        for text in legend.get_texts():
            text.set_fontweight('bold')  # Bold legend items

        # Customize the plot
        plt.xticks(rotation=45, ha='right', fontweight='bold')  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=12, fontweight='bold')
        plt.xlabel(taxa_lvl, fontsize=14, fontweight='bold')
        plt.ylabel('Relative Abundance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{path}/results_data/taxa_boxplot_Nextflow_QIIME_{taxa_lvl}.png", bbox_inches="tight", dpi=300)
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
            return best_matches

        if i % 2 == 1:  # If at species level
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
                metadata_columns = ['Sample_ID', 'location', 'sex', 'age', 'sample_type', 'breed', 'feeding-system',
                                    'temp', 'sequencing_method']

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
                group_values = [subset[subset[metadata_column.lower()] == group]["Relative_Abundance"] for group in
                                hue_groups]

                if len(group_values) == 2:  # Two groups → t-test
                    p_val = ttest_ind(group_values[0], group_values[1], equal_var=False).pvalue
                else:
                    p_val = None  # Extend to ANOVA/Kruskal-Wallis if more than two groups

                # Plot boxplot
                plt.figure(figsize=(4, 8))
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

                # Thicken the plot border (spines)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                # Customizations
                plt.xticks(rotation=45, ha='right', fontweight='bold')  # Rotate x-axis labels for better readability
                plt.yticks(fontsize=12, fontweight='bold')
                plt.xlabel(metadata_column, fontsize=14)
                plt.ylabel('Relative Abundance', fontsize=14)
                plt.title(f"{taxa_group_key}")
                plt.tight_layout()

                # Save the figure
                plt.savefig(f"{path}/results_data/boxplot_Nextflow_QIIME_{taxa_group_key}.png",
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

        # Loop through each matched taxon to create separate plots
        for taxon in matched_taxa:
            subset = long_df[long_df[taxa_lvl] == taxon]  # Filter data for this taxon

            if subset.empty:
                print(f"⚠️ Skipping {taxon}: No data available.")
                continue

            # Compute p-value for group differences
            hue_groups = subset[metadata_column.lower()].unique()
            group_values = [subset[subset[metadata_column.lower()] == group]["Relative_Abundance"] for group in
                            hue_groups]

            if len(group_values) == 2:  # Two groups → t-test
                p_val = ttest_ind(group_values[0], group_values[1], equal_var=False).pvalue
            else:
                p_val = None  # Extend to ANOVA/Kruskal-Wallis if more than two groups

            # Plot boxplot
            plt.figure(figsize=(4, 8))
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

            # Thicken the plot border (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            # Customizations
            plt.xticks(rotation=45, ha='right', fontweight='bold')  # Rotate x-axis labels for better readability
            plt.yticks(fontsize=12, fontweight='bold')
            plt.xlabel(metadata_column, fontsize=14, fontweight='bold')
            plt.ylabel('Relative Abundance', fontsize=14, fontweight='bold')
            plt.title(f"{taxon} - {taxa_lvl} Level")
            plt.tight_layout()

            # Save the figure
            plt.savefig(f"{path}/results_data/boxplot_{taxon.replace(' ', '_')}_Nextflow_QIIME_{taxa_lvl}.png",
                        bbox_inches="tight", dpi=300)
            plt.close()  # Close figure to prevent overlapping plots

        i += 1

    print("")
    print("Finished generating boxplots ٩(◕‿◕｡)۶")
    print("")

    def create_alpha_diversity_boxplot(path):
        print("")
        print("Generating alpha diversity boxplots...")

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
        alpha_metadata_df[location_col] = alpha_metadata_df[location_col].replace(
            {'arusha': 'Arusha', 'else where': 'Elsewhere'})

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
            ax.text(0.5, y_text, p_text, horizontalalignment='center', fontsize=12,
                    color='black')  # bold font: fontweight='bold'

            # Labels and title
            plt.xticks(rotation=45, ha='right', fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
            plt.xlabel("Location", fontweight='bold')
            plt.ylabel(metric.replace('_', ' ').title(), fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{path}/results_data/alpha_diversity_boxplot.png", dpi=300, bbox_inches="tight")

        print("Finsihed generating alpha diversity box plots ദ്ദി(｡•̀ ,<)~✩‧₊")


def create_alpha_diversity_boxplot(path):
    print("")
    print("Generating alpha diversity boxplots...")

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
    alpha_metadata_df[location_col] = alpha_metadata_df[location_col].replace(
        {'arusha': 'Arusha', 'else where': 'Elsewhere'})

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
        ax.text(0.5, y_text, p_text, horizontalalignment='center', fontsize=12,
                color='black')  # bold font: fontweight='bold'

        # Labels and title
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
        plt.xlabel("Location", fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title(), fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{path}/results_data/alpha_diversity_boxplot.png", dpi=300, bbox_inches="tight")

    print("Finsihed generating alpha diversity box plots ദ്ദി(｡•̀ ,<)~✩‧₊")