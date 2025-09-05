import process_data
import network_plots
import pcoa_plots
import pandas as pd


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
    if "Sample_ID" not in df1.columns or "Sample_ID" not in df2.columns:
        raise ValueError("Sample_ID column is missing from one of the datasets!")

    # Label the columns of each dataframe with "Bacteria_" or "Archaea_"
    columns_to_modify_ba = process_data.select_asv_columns(df1)  # Select ASV columns
    new_column_names_ba = {col: f"Bacteria_{col}" for col in columns_to_modify_ba}
    df1.rename(columns=new_column_names_ba, inplace=True)

    columns_to_modify_ar = process_data.select_asv_columns(df2)
    new_column_names_ar = {col: f"Archaea_{col}" for col in columns_to_modify_ar}
    df2.rename(columns=new_column_names_ar, inplace=True)

    # Identify metadata columns dynamically, ensuring ASV columns are excluded
    all_columns = set(df1.columns)
    non_asv_columns = all_columns - set(list(new_column_names_ba.values()))  # Exclude ASV columns
    metadata_cols = list(non_asv_columns - {"Sample_ID"})  # Ensure Sample_ID is not included

    # Merge on "Sample_ID" while keeping only one copy of the metadata columns
    df_combined = pd.merge(
        df1.drop(columns=metadata_cols, errors="ignore"),
        df2,  # Drop metadata from df2
        on="Sample_ID",
        how="inner"  # Ensures only matching samples are retained
    )

    return df_combined


def create_Ba_Ar_networks(ba_path, ar_path, path='.', with_labels=False):
    print("\nDoing combined Bacteria + Archaea analysis...\n")
    ############# PRE-PROCESSING #############

    ba_df_list = process_data.process_tables(ba_path)
    ar_df_list = process_data.process_tables(ar_path)

    # Combine Ba and Ar tables --> [Ba_Ar_genus_df, Ba_Ar_species_df]
    combined_df_list = []
    for i in range(len(ba_df_list)):
        combined_df = combine_datasets(ba_df_list[i], ar_df_list[i])
        combined_df_list.append(combined_df)

    ############# GENERATE NETWORKS #############

    network_plots.create_network_plots(path, combined_df_list, with_labels)

    return None


def create_Ba_Ar_pcoa_plot(ba_path, ar_path, path='.'):
    print("------ Processing combined Ba and Ar samples ------")

    ############# PRE-PROCESSING #############

    ba_df_list = process_data.process_tables(ba_path)
    ar_df_list = process_data.process_tables(ar_path)

    # Combine Ba and Ar tables --> [Ba_Ar_genus_df, Ba_Ar_species_df]
    combined_df_list = []
    for i in range(len(ba_df_list)):
        combined_df = combine_datasets(ba_df_list[i], ar_df_list[i])
        combined_df_list.append(combined_df)

    ############# GENERATE PCOA PLOTS #############

    pcoa_plots.create_pcoa_plot(path, combined_df_list)

    return None
