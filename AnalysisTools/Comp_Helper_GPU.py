import numpy as np
import cupy as cp
from numba import cuda
from sklearn.base import BaseEstimator, TransformerMixin
from cupyx.scipy.spatial.distance import pdist
from cuml import PCA, TruncatedSVD, TSNE, UMAP, KMeans, DBSCAN
from cuml.metrics import silhouette_score, silhouette_samples
from cuml.cluster import SpectralClustering
from cuml.manifold import MDS
from cupyx.scipy.sparse import csr_matrix
import cudf

class ComputeHelpersGPU:
    """
    A class containing helper functions for various computational tasks related to HiC data analysis,
    optimized for GPU execution using CUDA.
    """

    def __init__(self, memory_location: str = '.', memory_verbosity: int = 0):
        """
        Initialize the ComputeHelpersGPU class.

        Args:
            memory_location (str): The location for caching computations (not used in GPU version).
            memory_verbosity (int): The verbosity level for memory caching (not used in GPU version).
        """
        self.reduction_methods = {
            'pca': self._pca_reduction,
            'svd': self._svd_reduction,
            'tsne': self._tsne_reduction,
            'umap': self._umap_reduction,
            'mds': self._mds_reduction
        }
        
        self.clustering_methods = {
            'dbscan': self._dbscan_clustering,
            'spectral': self._spectral_clustering,
            'kmeans': self._kmeans_clustering
        }
        
        self.distance_metrics = {
            'euclidean': self.euclidean_distance,
            'pearsons': self.pearson_distance,
            'spearman': self.spearman_distance,
            'contact': self.contact_distance,
            "log2_contact": self.log2_contact_distance
        }
        
        self.normalization_methods = {
            'ice': self.ice_normalization,
            'log_transform': self.log_transform,
            'vc': self.vc_normalization,
            'kr': self.kr_norm,
        }

    '''#!========================================================== DATA LOADING AND PREPROCESSING ====================================================================================='''

    def getHiCData_simulation(self, filepath):
        """
        Load and process simulated HiC data on GPU.

        Args:
            filepath (str): Path to the input file.

        Returns:
            tuple: Processed HiC matrix, scaling vector, and error vector.
        """
        contactMap = cp.loadtxt(filepath)
        r = cp.triu(contactMap, k=1) 
        r = r / cp.max(r, axis=1, keepdims=True)
        rd = cp.transpose(r) 
        r = r + rd + cp.diag(cp.ones(len(r))) 

        D1 = []
        err = []
        for i in range(0, r.shape[0]): 
            D1.append((cp.mean(cp.diag(r,k=i)))) 
            err.append((cp.std(cp.diag(r,k=i))))
        
        return r, D1, err

    def getHiCData_experiment(self, filepath, cutoff=0.0, norm="max"):
        """
        Load and process experimental HiC data on GPU.

        Args:
            filepath (str): Path to the input file.
            cutoff (float): Cutoff value for filtering data.
            norm (str): Normalization method.

        Returns:
            tuple: Processed HiC matrix, scaling vector, and error vector.
        """
        contactMap = cp.loadtxt(filepath)
        r = cp.triu(contactMap, k=1)
        r[cp.isnan(r)] = 0.0
        r = r / cp.max(r, axis=1, keepdims=True)
        
        if norm == "first":
            for i in range(len(r) - 1):
                maxElem = r[i][i + 1]
                if(maxElem != cp.max(r[i])):
                    for j in range(len(r[i])):
                        if maxElem != 0.0:
                            r[i][j] = float(r[i][j] / maxElem)
                        else:
                            r[i][j] = 0.0 
                        if r[i][j] > 1.0:
                            r[i][j] = .5
        r[r<cutoff] = 0.0
        rd = cp.transpose(r) 
        r = r + rd + cp.diag(cp.ones(len(r)))

        D1 = []
        err = []
        for i in range(0, r.shape[0]): 
            D1.append((cp.mean(cp.diag(r,k=i)))) 
            err.append((cp.std(cp.diag(r,k=i))))
        D = cp.array(D1)
        err = cp.array(err)

        return r, D, err

    def pad_array(self, array, target_shape):
        """
        Pad a 3D array to the target shape with zeros on GPU.

        Args:
            array (cp.array): The array to pad.
            target_shape (tuple): The target shape to pad to.

        Returns:
            cp.array: The padded array.
        """
        result = cp.zeros(target_shape)
        slices = tuple(slice(0, dim) for dim in array.shape)
        result[slices] = array
        return result
    
    '''#!========================================================== DISTANCE CALCULATIONS ====================================================================================='''

    def calc_dist(self, X: cp.array, metric: str) -> cp.array:
        """
        Calculate the distance matrix using the specified metric on GPU.

        Args:
            X (cp.array): The input data matrix.
            metric (str): The distance metric to use.

        Returns:
            cp.array: The calculated distance matrix.
        """
        try:
            return self.distance_metrics[metric](X)
        except KeyError:
            raise KeyError(f"Invalid metric: {metric}. Available metrics are: {list(self.distance_metrics.keys())}")

    def euclidean_distance(self, x):
        """Calculate the Euclidean distance matrix on GPU."""
        return squareform(pdist(x, metric='euclidean'))

    def contact_distance(self, x):
        """Calculate the contact distance matrix on GPU."""
        return squareform(pdist(x, metric='cityblock'))

    def pearson_distance(self, X: cp.array) -> cp.array:
        """Calculate Pearson correlation distance on GPU."""
        corr = cp.corrcoef(X)
        return 1 - corr

    def spearman_distance(self, X: cp.array) -> cp.array:
        """Calculate Spearman correlation distance on GPU."""
        rank_data = cp.apply_along_axis(lambda x: cp.argsort(cp.argsort(x)), 1, X)
        return 1 - cp.corrcoef(rank_data)

    def log2_contact_distance(self, X: cp.array) -> cp.array:
        """Calculate log2 contact distance on GPU."""
        log2_X = cp.log2(X + 1)
        return squareform(pdist(log2_X, metric='cityblock'))
    
    '''#!========================================================== NORMALIZATION METHODS ====================================================================================='''

    def norm_distMatrix(self, matrix: cp.array, norm: str):
        """
        Normalize the distance matrix using the specified method on GPU.

        Args:
            matrix (cp.array): The input distance matrix.
            norm (str): The normalization method to use.

        Returns:
            cp.array: The normalized distance matrix.
        """        
        try:
            return self.normalization_methods[norm](matrix)
        except KeyError:
            raise KeyError(f"Invalid normalization method: {norm}. Available methods are: {list(self.normalization_methods.keys())}")

    def log_transform(self, m):
        """Perform log transformation on GPU."""
        return cp.log2(m + 1)

    def vc_normalization(self, m):
        """Perform variance stabilizing normalization on GPU."""
        return m / m.sum(axis=1, keepdims=True)

    @cuda.jit
    def _ice_normalization_kernel(matrix, bias, max_iter, tolerance):
        """CUDA kernel for ICE normalization."""
        i, j = cuda.grid(2)
        if i < matrix.shape[0] and j < matrix.shape[1]:
            for _ in range(max_iter):
                old_bias = bias[i]
                row_sum = cuda.atomic.add(cuda.local.array(1, dtype=cp.float64), 0, matrix[i, j])
                col_sum = cuda.atomic.add(cuda.local.array(1, dtype=cp.float64), 0, matrix[j, i])
                if row_sum != 0 and col_sum != 0:
                    bias[i] *= cp.sqrt(row_sum * col_sum)
                else:
                    bias[i] = 1
                if bias[i] != 0 and bias[j] != 0:
                    matrix[i, j] /= (bias[i] * bias[j])
                else:
                    matrix[i, j] = 0
                if abs(bias[i] - old_bias) < tolerance:
                    break

    def ice_normalization(self, matrix: cp.array, max_iter: int=100, tolerance: float=1e-5) -> cp.array:
        """Perform ICE normalization on GPU."""
        n = matrix.shape[0]
        bias = cp.ones(n)
        matrix_balanced = matrix.copy()
        
        threadsperblock = (16, 16)
        blockspergrid_x = (matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        self._ice_normalization_kernel[blockspergrid, threadsperblock](matrix_balanced, bias, max_iter, tolerance)
        
        return matrix_balanced

    def kr_norm(self, matrix: cp.array, max_iter: int=100, tolerance: float=1e-5) -> cp.array:
        """Perform KR normalization on GPU."""
        # Implement KR normalization using CUDA kernel (similar to ICE normalization)
        # This is a placeholder and needs to be implemented
        return matrix
    
    '''#!========================================================== DIMENSIONALITY REDUCTION METHODS ====================================================================================='''

    def run_reduction(self, method, X, n_components):
        """
        Run the specified dimensionality reduction method on GPU.

        Args:
            method (str): The reduction method to use.
            X (cp.array): Input data.
            n_components (int): Number of components for the reduction.

        Returns:
            tuple: Reduced data and additional information (if available).
        """
        try:
            return self.reduction_methods[method](X, n_components)
        except KeyError:
            raise KeyError(f"Invalid reduction method: {method}. Available methods are: {list(self.reduction_methods.keys())}")

    def _pca_reduction(self, X, n_components):
        """Perform PCA reduction on GPU."""
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(X)
        return result, pca.explained_variance_ratio_, pca.components_

    def _svd_reduction(self, X, n_components):
        """Perform SVD reduction on GPU."""
        svd = TruncatedSVD(n_components=n_components)
        result = svd.fit_transform(X)
        return result

    def _tsne_reduction(self, X, n_components):
        """Perform t-SNE reduction on GPU."""
        tsne = TSNE(n_components=n_components)
        result = tsne.fit_transform(X)
        return result, None, None  # TSNE in cuML doesn't provide kl_divergence

    def _umap_reduction(self, X, n_components):
        """Perform UMAP reduction on GPU."""
        umap_reducer = UMAP(n_components=n_components)
        result = umap_reducer.fit_transform(X)
        return result, None, None  # UMAP in cuML doesn't provide embedding_ and graph_

    def _mds_reduction(self, X, n_components):
        """Perform MDS reduction on GPU."""
        mds = MDS(n_components=n_components)
        result = mds.fit_transform(X)
        return result, None, None  # MDS in cuML doesn't provide stress_ and dissimilarity_matrix_

    '''#!========================================================== CLUSTERING METHODS ====================================================================================='''

    def run_clustering(self, method, X, **kwargs):
        """
        Run the specified clustering method on GPU.

        Args:
            method (str): The clustering method to use.
            X (cp.array): Input data.
            **kwargs: Additional arguments for the clustering method.

        Returns:
            cp.array: Cluster labels.
        """
        try:
            return self.clustering_methods[method](X, **kwargs)
        except KeyError:
            raise KeyError(f"Invalid clustering method: {method}. Available methods are: {list(self.clustering_methods.keys())}")

    def _kmeans_clustering(self, X: cp.array, n_clusters: int, **kwargs):
        """Perform K-means clustering on GPU."""
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        return kmeans.fit_predict(X)

    def _spectral_clustering(self, X, n_clusters, **kwargs):
        """Perform Spectral clustering on GPU."""
        sc = SpectralClustering(n_clusters=n_clusters, **kwargs)
        return sc.fit_predict(X)

    def _dbscan_clustering(self, X, eps, min_samples, **kwargs):
        """Perform DBSCAN clustering on GPU."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        return dbscan.fit_predict(X)

    def find_optimal_clusters(self, data: cp.array, max_clusters: int=10):
        """
        Find the optimal number of clusters using the elbow method and silhouette score on GPU.

        Args:
            data (cp.array): Input data for clustering.
            max_clusters (int): Maximum number of clusters to consider.

        Returns:
            int: Optimal number of clusters.
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
        
        # Use the elbow method to find the optimal number of clusters
        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        elbow = kl.elbow if kl.elbow else max_clusters
        
        # Find the number of clusters with the highest silhouette score
        silhouette_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Return the smaller of the two to be conservative
        return min(elbow, silhouette_optimal)

    def evaluate_clustering(self, data, labels):
        """
        Evaluate clustering quality using various metrics on GPU.

        Args:
            data (cp.array): Input data used for clustering.
            labels (cp.array): Cluster labels assigned to the data points.

        Returns:
            tuple: Silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
        """
        silhouette = silhouette_score(data, labels)
        
        # Note: cuML doesn't provide Calinski-Harabasz and Davies-Bouldin indices
        # We'll use placeholder values and print a warning
        calinski_harabasz = None
        davies_bouldin = None
        
        print("\nClustering Evaluation Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print("  Calinski-Harabasz Index: Not available in GPU implementation")
        print("  Davies-Bouldin Index: Not available in GPU implementation")
        
        return silhouette, calinski_harabasz, davies_bouldin

    """==================================================================== UTILITY METHODS =========================================================="""

    def _calc_dist_wrapper(self, trajectories, metric):
        """
        Wrapper function for calc_dist to be used with GPU computation.

        Args:
            trajectories (list): List of trajectory data.
            metric (str): Distance metric to use.

        Returns:
            list: List of distance matrices.
        """
        return [self.calc_dist(cp.array(val), metric) for val in trajectories]

    def cached_calc_dist(self, trajectories, metric):
        """
        Calculates distance matrices on GPU (caching not implemented for GPU version).

        Args:
            trajectories (list): List of trajectory data.
            metric (str): Distance metric to use.

        Returns:
            list: List of distance matrices.
        """
        return self._calc_dist_wrapper(trajectories, metric)

    def getNormMethods(self):
        """Get the list of available normalization methods."""
        return list(self.normalization_methods)
    
    def getDistanceMetrics(self):
        """Get the list of available distance metrics."""
        return list(self.distance_metrics)
    
    def getClusteringMethods(self):
        """Get the list of available clustering methods."""
        return list(self.clustering_methods)
    
    def getReductionMethods(self):
        """Get the list of available dimensionality reduction methods."""
        return list(self.reduction_methods)
    
    def getMemStats(self):
        """Get memory statistics (not applicable for GPU version)."""
        return None
    
    def setMem(self, path: str, verbose: int = 0):
        """Set memory location and verbosity (not applicable for GPU version)."""
        pass  # No-op for GPU version

    """==================================================================== GPU-SPECIFIC UTILITY METHODS =========================================================="""

    def to_gpu(self, data):
        """
        Convert numpy array or list to CuPy array.

        Args:
            data: Input data (numpy array or list).

        Returns:
            cp.array: CuPy array.
        """
        if isinstance(data, np.ndarray):
            return cp.array(data)
        elif isinstance(data, list):
            return cp.array(np.array(data))
        else:
            raise TypeError("Input must be a numpy array or a list")

    def to_cpu(self, data):
        """
        Convert CuPy array to numpy array.

        Args:
            data: Input data (CuPy array).

        Returns:
            np.array: Numpy array.
        """
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        else:
            raise TypeError("Input must be a CuPy array")

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
    """================================================== OTHER UTILS ==================================================================="""
    def squareform(distances):
        n = int(np.sqrt(len(distances) * 2)) + 1
        dist_matrix = cp.zeros((n, n))
        dist_matrix[cp.triu_indices(n, k=1)] = distances
        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix