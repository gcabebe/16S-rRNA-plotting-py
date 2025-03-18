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

# Setup
The python code is run through a shell script. If using Windows, download [Git for Windows](https://gitforwindows.org/).

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

# Getting Started

