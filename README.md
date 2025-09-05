# Summary
This repository provides an executable script for generating visualizations from processed amplicon data extracted from bacteria and archaea. Visualizations will analyze each bacteria and archaea in isolation, as well as the interactions between the two.

**Inputs**
- ASV table (1 each for bacteria and archaea)
- Taxonomy classification table (1 each for bacteria and archaea)
- Metadata table

**Outputs**
-  Network plots on the genus and species level
-  Table of global network properties
-  Table of network topology
-  Taxonomy boxplot
-  Alpha diversity metrics boxplot
-  PcoA plots using Bray-Curtis distances

---

# Setup
The python code is run through a shell script. If using Windows, use Windows PowerShell or download [Git for Windows](https://gitforwindows.org/). For Mac users use terminal.

The following Python 3 libraries are required:
- pandas
- NumPy
- matplotlib
- seaborn
- scipy
- scikit-learn
- networkx
- markov_clustering
- rapidfuzz
- ete3

---

# Quick Start
This code assumes the following directory tree structure and file names:
```
.
└── Root_Folder/
    ├── metadata.tsv
    ├── Bacteria_Data/
    │   ├── feature-table.tsv
    │   ├── taxonomy.tsv
    │   ├── qiime_genus.tsv
    │   └── qiime_specie.tsv
    └── Archaea_Data/
        ├── feature-table.tsv
        ├── taxonomy.tsv
        ├── qiime_genus.tsv
        └── qiime_specie.tsv
```

In your terminal run the following:

```./driver_script.sh <path_to_root_folder> <bacteria_folder_name> <archaea_folder_name> ```

- `<path_to_root_folder>`: Absolute or relative path to the root data folder.
- `<bacteria_folder_name>`: Name of the folder containing bacterial data (e.g., `Bacteria_Data`).
- `<archaea_folder_name>`: Name of the folder containing archaeal data (e.g., `Archaea_Data`).
