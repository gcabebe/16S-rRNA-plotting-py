from process_data import select_asv_columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, mannwhitneyu, permutation_test
from sklearn.decomposition import PCA


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
    """Generates a Bray-Curtis PCoA plot and performs PERMANOVA using SciPy."""
    permanova_results = []  # Store results for CSV

    i = 0
    for df in df_list:  # [genus_df, species_df]
        taxa_lvl = "Genus" if i % 2 == 0 else "Species"

        # ---- Extract ASV data ----
        asv_data_columns = select_asv_columns(df)
        asv_data = df[asv_data_columns].fillna(0)

        # ---- Compute Bray-Curtis Distance Matrix ----
        bc_dist = squareform(pdist(asv_data, metric="braycurtis"))

        # ---- Perform PCoA ----
        pcoa = PCA(n_components=2)
        pcoa_results = pcoa.fit_transform(bc_dist)

        # Get variance explained
        variance_explained = pcoa.explained_variance_ratio_ * 100
        pc1_var = round(variance_explained[0], 2)
        pc2_var = round(variance_explained[1], 2)

        # Convert to DataFrame
        pcoa_df = pd.DataFrame(pcoa_results, columns=["PCoA1", "PCoA2"], index=df.index)
        pcoa_df["location"] = df["location"].values  # Add location info

        # ---- Set Color Palette ----
        unique_locations = pcoa_df["location"].unique()
        palette_colors = sns.color_palette(palette, len(unique_locations))
        color_dict = dict(zip(unique_locations, palette_colors))  # Map locations to colors

        # ---- Plot PCoA ----
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        sns.scatterplot(data=pcoa_df, x="PCoA1", y="PCoA2", hue="location", s=100, edgecolor="black",
                        palette=color_dict)

        for location, group in pcoa_df.groupby("location"):
            draw_ellipse(group["PCoA1"], group["PCoA2"], ax, color=color_dict[location])

        plt.xlabel(f"PCoA1 ({pc1_var}%)", fontweight='bold')
        plt.ylabel(f"PCoA2 ({pc2_var}%)", fontweight='bold')
        plt.title(f"PCoA Plot (Bray-Curtis Distance) - {taxa_lvl} Level", fontweight='bold')
        plt.legend(title="Location")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/results_data/pcoa_bray_curtis_{taxa_lvl}.png", dpi=300, bbox_inches="tight")

        # ---- Perform PERMANOVA Using SciPy ----
        labels = df["location"].values
        unique_labels = np.unique(labels)

        # Compute mean within-group and between-group distances
        def compute_statistic(labels):
            within_group_dists = []
            between_group_dists = []

            for label in unique_labels:
                mask = labels == label
                within = bc_dist[np.ix_(mask, mask)]
                if within.size > 0:
                    within_group_dists.append(np.mean(within))

            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i < j:  # Avoid duplicate pairs
                        mask1, mask2 = labels == label1, labels == label2
                        between = bc_dist[np.ix_(mask1, mask2)]
                        if between.size > 0:
                            between_group_dists.append(np.mean(between))

            return np.mean(between_group_dists) - np.mean(within_group_dists)

        observed_stat = compute_statistic(labels)

        perm_result = permutation_test(
            (labels,),
            statistic=compute_statistic,
            permutation_type="pairings",
            alternative="two-sided",
            n_resamples=999
        )

        # Store PERMANOVA results
        permanova_results.append({
            "Taxa Level": taxa_lvl,
            "Pseudo-F": observed_stat,  # Approximate test statistic
            "p-value": perm_result.pvalue,
            "Permutations": 999
        })

        i += 1

    # Save PERMANOVA results to CSV
    permanova_df = pd.DataFrame(permanova_results)
    permanova_df.to_csv(f"{path}/results_data/permanova_results.csv", index=False)

    print("Done creating PCoA plots and PERMANOVA results saved ( ˶ˆᗜˆ˵ ) \n")
