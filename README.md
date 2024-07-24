# OpenMiChroM-Ana

OpenMiChroM-Ana is a Python package for analyzing chromosome structure data. It provides tools for data processing, dimensionality reduction, clustering, and visualization of Hi-C and related genomic data. The package supports both CPU and GPU acceleration for computationally intensive tasks.

## Features

- Data loading and preprocessing for Hi-C and simulated data
- Distance matrix calculations with various metrics
- Normalization methods (ICE, KR, VC, log transform)
- Dimensionality reduction techniques (PCA, SVD, t-SNE, UMAP, MDS)
- Clustering algorithms (K-means, DBSCAN, Spectral, Hierarchical, OPTICS)
- Clustering evaluation metrics
- GPU acceleration for supported operations
- Visualization tools for analysis results

## Installation

### Prerequisites
- Python 3.10
- For GPU support: CUDA toolkit 11.8 or higher


### CPU Version

To install the CPU-only version of OpenMiChroM-Ana, run:

```bash
pip install .
```

### GPU Version

The GPU version requires CUDA and the RAPIDS suite of libraries. Follow these steps:
- Install CUDA toolkit
- create new conda or venv environment

```bash
conda create -n [envName] python=3.11
conda activate [envName]
```

- Install RAPIDS Library
```bash
conda install -c rapidsai -c nvidia -c conda-forge cudf=23.08 cuml=23.08 cugraph=23.08 cudatoolkit=11.8
```

- Install OpenMiChroM-Ana with GPU support:
```bash
pip install .[gpu]
```


## Usage
Here's a basic example of how to use OpenMiChroM-Ana:

### For CPU Usage
```python
from OpenMiChroM_Ana import Ana

# Initialize the analysis object
analysis = Ana(showPlots=True, execution_mode='cpu', cacheStoragePath='/path/to/cache')
```
### For GPU Usage
```python
from OpenMiChroM_Ana import Ana

# Initialize the analysis object
analysis = Ana(showPlots=True, execution_mode='gpu', cacheStoragePath='/path/to/cache')
```

### Example
```python
# Add datasets
analysis.add_dataset(label="Dataset1", folder="data/Dataset1")
analysis.add_dataset(label='Dataset2', folder='data/Dataset2')

# Process trajectories
analysis.process_trajectories(label="Dataset1", filename="traj_1.cndb", folder_pattern=['iteration_', [1, 20]])
analysis.process_trajectories(label="Dataset2", filename="traj_2.cndb", folder_pattern=['iteration_', [1, 20]])

# Perform PCA
pca_result = analysis.pca("Dataset1", "Dataset2", metric='euclidean', n_components=2, norm='ice', method='weighted')

# Perform clustering
kmeans_result = analysis.kmeans_clustering("Dataset1", "Dataset2", n_clusters=5, metric='euclidean', norm='ice', method='weighted')
# Generate plots
# Plots will be saved automatically if showPlots=True
```
### Contributing
Contributions to OpenMiChroM-Ana are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
If you have any questions or feedback, please open an issue on the GitHub repository.
