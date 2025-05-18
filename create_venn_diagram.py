
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def create_venn_diagram(path, group_col='location'):
    print(f"Creating Venn diagram for metadata column: '{group_col}'")

    # Read and process taxonomy
    taxa_df = pd.read_csv(f"{path}/taxonomy.tsv", sep='\t', skiprows=[1])
    taxonomy_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    taxa_df[taxonomy_levels] = taxa_df["Taxon"].str.extract(
        r'd__(?P<Kingdom>[^;]*)'
        r'(?:; p__(?P<Phylum>[^;]*))?'
        r'(?:; c__(?P<Class>[^;]*))?'
        r'(?:; o__(?P<Order>[^;]*))?'
        r'(?:; f__(?P<Family>[^;]*))?'
        r'(?:; g__(?P<Genus>[^;]*))?'
        r'(?:; s__(?P<Species>[^;]*))?'
    )
    taxa_df = taxa_df.drop(columns=['Taxon'])

    # Read ASV and metadata tables
    asv_df = pd.read_csv(f"{path}/feature-table.tsv", sep='\t', skiprows=1)
    metadata_df = pd.read_csv("metadata.tsv", sep='\t')
    metadata_df.rename(columns={metadata_df.columns[0]: "Sample_ID"}, inplace=True)

    # Melt and merge
    asv_melted = asv_df.melt(id_vars='#OTU ID', var_name='Sample_ID', value_name='Abundance')
    merged_df = pd.merge(asv_melted, taxa_df, left_on='#OTU ID', right_on='Feature ID')
    df = pd.merge(merged_df, metadata_df, on='Sample_ID')

    # Check that the column exists
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' is not a valid metadata column.")

    # Get unique groups
    groups = df[group_col].dropna().unique()
    if len(groups) < 2 or len(groups) > 3:
        raise ValueError("Venn diagrams require exactly 2 or 3 groups for comparison.")

    # Create sets of ASVs for each group
    group_sets = {}
    for group in groups:
        asv_ids = df.loc[(df[group_col] == group) & (df['Abundance'] > 0), "#OTU ID"]
        group_sets[group] = set(asv_ids)

    # Plot
    plt.figure(figsize=(6, 6))
    if len(groups) == 2:
        g1, g2 = groups
        venn2([group_sets[g1], group_sets[g2]], set_labels=(g1, g2))
    elif len(groups) == 3:
        g1, g2, g3 = groups
        venn3([group_sets[g1], group_sets[g2], group_sets[g3]], set_labels=(g1, g2, g3))

    plt.title(f"Venn Diagram of ASVs by {group_col.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{path}/results_data/venn_diagram_{group_col}.png", dpi=300, bbox_inches="tight")

    print(f"Finished creating venn diagrams for the '{group_col}' metadata column")