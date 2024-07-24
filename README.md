# OpenMiChroM-Ana: Advanced Chromosome Structure Analysis Tool

## Overview

OpenMiChroM-Ana is a powerful Python package designed for comprehensive analysis of chromosome structure data. It offers a suite of tools for processing, analyzing, and visualizing Hi-C and related genomic data. With support for both CPU and GPU acceleration, OpenMiChroM-Ana provides researchers with a flexible and efficient platform for exploring complex genomic structures.
## Key Features

- Versatile Data Handling:
  * Support for Hi-C and simulated chromosome structure data
  * Efficient loading and preprocessing capabilities

- Comprehensive Analysis Tools:
  * Distance matrix calculations with multiple metrics
  * Advanced normalization methods (ICE, KR, VC, log transform)
  * State-of-the-art dimensionality reduction techniques (PCA, SVD, t-SNE, UMAP, MDS)
  * Diverse clustering algorithms (K-means, DBSCAN, Spectral, Hierarchical, OPTICS)
  * Robust clustering evaluation metrics

- Performance Optimization:
  * GPU acceleration for computationally intensive operations
  * Efficient CPU implementations for broad compatibility

- Visualization:
  * Rich set of plotting tools for result interpretation
  * Interactive visualizations for in-depth data exploration
  * 


## Installation
### System Requirements

- Python 3.11 or higher
- For GPU support: CUDA version 12.0 or higher

### CPU Version Installation

For users who prefer CPU-parallel operations:

```bash
# Navigate to the directory where the pyproject.toml is placed
pip install .
```

### GPU Enabled Version
To leverage GPU acceleration:
1. Install CUDA Libraries (version 12.0 or higher)
2. Set up conda or virtual env
3. Install RAPIDS Suite (follow instructions at https://docs.rapids.ai/install)

```bash
conda create -n [envName] -c rapidsai -c conda-forge -c nvidia rapids=24.06 python=3.11 cuda-version=12.0
conda activate [envName]
# Navigate to the directory where the pyproject.toml is placed
pip install .[gpu]
```


## Quick Start Guide
### Initializing the Analysis
For CPU usage:
```python
from OpenMiChroM_Ana import Ana

analysis = Ana(showPlots=True, execution_mode='cpu', cacheStoragePath='/path/to/cache')
```
For GPU usage:
```python
from OpenMiChroM_Ana import Ana

analysis = Ana(showPlots=True, execution_mode='gpu', cacheStoragePath='/path/to/cache')
```
### Basic Workflow Example
```python
# Load datasets
analysis.add_dataset(label="ExperimentA", folder="data/ExperimentA")
analysis.add_dataset(label="ExperimentB", folder="data/ExperimentB")

# Process trajectory data
analysis.process_trajectories(label="ExperimentA", filename="traj_A.cndb", folder_pattern=['iteration_', [1, 20]])
analysis.process_trajectories(label="ExperimentB", filename="traj_B.cndb", folder_pattern=['iteration_', [1, 20]])

# Perform dimensionality reduction
pca_result = analysis.pca("ExperimentA", "ExperimentB", metric='euclidean', n_components=2, norm='ice', method='weighted')

# Conduct clustering analysis
kmeans_result = analysis.kmeans_clustering("ExperimentA", "ExperimentB", n_clusters=5, metric='euclidean', norm='ice', method='weighted')

# Visualize results
# Plots are automatically saved if showPlots=True
```


## Contribution
We welcome contributions to OpenMiChroM-Ana! Whether it's bug fixes, feature additions, or documentation improvements, your input is valuable. Please review our contribution guidelines before submitting a pull request.



## License
OpenMiChroM-Ana is distributed under the MIT License. See the LICENSE file in the repository for full details.


## Support and Contact
For bug reports and feature requests, please use the GitHub issue tracker. 