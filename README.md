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

### CPU Version

To install the CPU-only version of OpenMiChroM-Ana, run:

```bash
pip install .
```

### GPU Version

To install the GPU-acceleratred version (requires CUDA-enabled GPU), run:

```bash
pip install .[gpu]
```
Note: Make sure you have the appropiate CUDA Toolkit isntalled for GPU support.

### Requirements
<ol>
    <li> Python 3.6+ </li>
    <li> numpy </li>
    <li> scipy </li>
    <li> scikit-learn </li>
    <li> joblib </li>
    <li> numba </li>
    <li> matplotlib </li>
    <li> seaborn </li>
    <li> umap-learn </li>
    <li> kneed </li>
    <li> pandas </li>
</ol>

### Additional requirements for GPU support

<ol>
    <li> cupy </li>
    <li> cuml </li>
    <li> cugraph </li>
    <li> cudf </li>
</ol>

### Usage
Here's a basic example of how to use OpenMiChroM-Ana:

```python
from OpenMiChroM_Ana import Ana

# Initialize the analysis object
analysis = Ana(showPlots=True, execution_mode='gpu', cacheStoragePath='/path/to/cache')

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
