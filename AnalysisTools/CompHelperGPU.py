import cupy as cp
import numpy as np
from numba import cuda
from cuml import PCA, TruncatedSVD, TSNE, UMAP, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
# from cuml.manifold import MDS
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.spatial.distance import pdist
import cudf
from kneed import KneeLocator


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

@cuda.jit
def _kr_normalization_kernel(matrix, bias, max_iter, tolerance):
    """CUDA kernel for KR normalization."""
    i, j = cuda.grid(2)
    if i < matrix.shape[0] and j < matrix.shape[1]:
        for _ in range(max_iter):
            old_bias = bias[i]
            row_sum = cuda.atomic.add(cuda.local.array(1, dtype=cp.float64), 0, matrix[i, j])
            col_sum = cuda.atomic.add(cuda.local.array(1, dtype=cp.float64), 0, matrix[j, i])
            if bias[i] != 0:
                bias[i] *= cp.sqrt(row_sum * col_sum)
            else:
                bias[i] = 1
            if bias[i] != 0 and bias[j] != 0:
                matrix[i, j] = matrix[i, j] / (bias[i] * bias[j])
            else:
                matrix[i, j] = 0
            if abs(bias[i] - old_bias) < tolerance:
                break


@cuda.jit
def _pearson_distance_kernel(X, result):
    i, j = cuda.grid(2)
    if i < X.shape[0] and j < X.shape[0]:
        if i != j:
            mean_i = 0.0
            mean_j = 0.0
            for k in range(X.shape[1]):
                mean_i += X[i, k]
                mean_j += X[j, k]
            mean_i /= X.shape[1]
            mean_j /= X.shape[1]

            numerator = 0.0
            denom_i = 0.0
            denom_j = 0.0
            for k in range(X.shape[1]):
                diff_i = X[i, k] - mean_i
                diff_j = X[j, k] - mean_j
                numerator += diff_i * diff_j
                denom_i += diff_i * diff_i
                denom_j += diff_j * diff_j

            denominator = cp.sqrt(denom_i * denom_j)
            if denominator != 0:
                result[i, j] = 1 - (numerator / denominator)
            else:
                result[i, j] = 1.0  # Maximum distance when denominator is zero
class ComputeHelpersGPU:
    """
    A class containing helper functions for various computational tasks related to HiC data analysis,
    optimized for GPU execution using CUDA.
    """

    def __init__(self, memory_location: str = '.', memory_verbosity: int = 0):
        """__init__
        
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
            # 'mds': self._mds_reduction
        }
        
        self.clustering_methods = {
            'dbscan': self._dbscan_clustering,
            # 'spectral': self._spectral_clustering,
            'kmeans': self._kmeans_clustering,
            'hierarchical': self._hierarchical_clustering,
            'optics': self._optics_clustering
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
        return self.squareform(pdist(x, metric='euclidean'))
    
    def contact_distance(self, x):
        """Calculate the contact distance matrix on GPU."""
        return self.squareform(pdist(x, metric='cityblock'))

    def pearson_distance(self, X: cp.array) -> cp.array:
        """Calculate Pearson correlation distance on GPU."""
        result = cp.zeros((X.shape[0], X.shape[0]), dtype=cp.float32)
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (X.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (X.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
        )
        _pearson_distance_kernel[blocks_per_grid, threads_per_block](X, result)
        return result

    def spearman_distance(self, X: cp.array) -> cp.array:
        """Calculate Spearman correlation distance on GPU."""
        rank_data = cp.apply_along_axis(lambda x: cp.argsort(cp.argsort(x)), 1, X)
        return self.pearson_distance(rank_data)
    
    def log2_contact_distance(self, X: cp.array) -> cp.array:
        """Calculate log2 contact distance on GPU."""
        epsilon = cp.finfo(float).eps
        log2_X = cp.log2(X + epsilon)
        log2_X[~cp.isfinite(log2_X)] = 0
        return self.squareform(pdist(log2_X, metric='cityblock'))
    
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
        return self.safe_divide(m, m.sum(axis=1, keepdims=True))



    def ice_normalization(self, matrix: cp.array, max_iter: int=100, tolerance: float=1e-5) -> cp.array:
        """Perform ICE normalization on GPU."""
        n = matrix.shape[0]
        bias = cp.ones(n)
        matrix_balanced = matrix.copy()
        
        threadsperblock = (16, 16)
        blockspergrid_x = (matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        _ice_normalization_kernel[blockspergrid, threadsperblock](matrix_balanced, bias, max_iter, tolerance)
        
        return matrix_balanced



    def kr_norm(self, matrix: cp.array, max_iter: int=100, tolerance: float=1e-5) -> cp.array:
        """Perform KR normalization on GPU."""
        n = matrix.shape[0]
        bias = cp.ones(n)
        matrix_balanced = matrix.copy()
        
        threadsperblock = (16, 16)
        blockspergrid_x = (matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        _kr_normalization_kernel[blockspergrid, threadsperblock](matrix_balanced, bias, max_iter, tolerance)
        
        return matrix_balanced
    
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
            return self.reduction_methods[method](X, n_components, **kwargs)
        except KeyError:
            raise KeyError(f"Invalid reduction method: {method}. Available methods are: {list(self.reduction_methods.keys())}")

    def _pca_reduction(self, X, n_components):
        """Perform PCA reduction on GPU."""
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(X)
        feature_importance = cp.abs(pca.components_[0])
        sorted_idx = cp.argsort(feature_importance)
        print("Feature Importance: ")
        for idx in sorted_idx[-10:]: # print top 10 features
            print(f"Feature {idx}: {feature_importance[idx]:.4f}")
        return result, pca.explained_variance_ratio_, pca.components_

    def _svd_reduction(self, X, n_components):
        """Perform SVD reduction on GPU."""
        svd = TruncatedSVD(n_components=n_components)
        result = svd.fit_transform(X)
        return result, svd.singular_values_, svd.components_

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

    # def _mds_reduction(self, X, n_components):
    #     """Perform MDS reduction on GPU."""
    #     mds = MDS(n_components=n_components)
    #     result = mds.fit_transform(X)
    #     return result, None, None  # MDS in cuML doesn't provide stress_ and dissimilarity_matrix_

    '''#!========================================================== CLUSTERING METHODS ====================================================================================='''

    def run_clustering(self, method, X, n_components: int, n_clusters: int, **kwargs):
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
            return self.clustering_methods[method](X, n_clusters=n_clusters, n_components=n_components, **kwargs)
        except KeyError:
            raise KeyError(f"Invalid clustering method: {method}. Available methods are: {list(self.clustering_methods.keys())}")

    def _kmeans_clustering(self, X: cp.array, n_clusters: int, **kwargs):
        """Perform K-means clustering on GPU."""
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        labels = kmeans.fit_predict(X)
        inertias = []
        for k in range(1, n_clusters):
            kmeans_temp = KMeans(n_clusters=k)
            kmeans_temp.fit(X)
            inertias.append(kmeans_temp.inertia_)
        return labels, {
            'inertia': inertias,
            'cluster_centers': kmeans.cluster_centers_,
            'n_iter': kmeans.n_iter_
        }
        
    # def _spectral_clustering(self, X, n_clusters, **kwargs):
    #     """Perform Spectral clustering on GPU."""
    #     spectral = SpectralClustering(n_clusters=n_clusters, n_components=n_components, **kwargs)
    #     labels = spectral.fit_predict(X)
    #     return labels, {
    #         'affinity_matrix_': spectral.affinity_matrix_,
    #         'n_features_in': spectral.n_features_in_
    #     }


    def _dbscan_clustering(self, X, eps, min_samples, **kwargs):
        """Perform DBSCAN clustering on GPU."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(X)
        return labels, {
            'core_sample_indices': dbscan.core_sample_indices_,
            'components': dbscan.components_,
            'n_features_in': dbscan.n_features_in_
        }
        
    def _hierarchical_clustering(self, X, n_clusters, **kwargs):
        # Note: Hierarchical clustering is not available in cuML. 
        # We'll need to move data to CPU, perform clustering, and move back to GPU.
        from scipy.cluster.hierarchy import linkage, fcluster
        X_cpu = cp.asnumpy(X)
        linkage_matrix = linkage(X_cpu, method=kwargs.get('method', 'ward'))
        labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        return cp.array(labels), {
            'linkage_matrix': cp.array(linkage_matrix)
        }
        
    
    def _optics_clustering(self, X, min_samples, xi, min_cluster_size, **kwargs):
        # Note: OPTICS is not available in cuML. 
        # We'll need to move data to CPU, perform clustering, and move back to GPU.
        from sklearn.cluster import OPTICS
        X_cpu = cp.asnumpy(X)
        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, **kwargs)
        labels = optics.fit_predict(X_cpu)
        return cp.array(labels), {
            'reachability': cp.array(optics.reachability_),
            'ordering': cp.array(optics.ordering_),
            'core_distances': cp.array(optics.core_distances_),
            'predecessor': cp.array(optics.predecessor_)
        }


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

    def evaluate_clustering(self, data, cluster_labels):
        """
        Evaluate clustering quality using various metrics on GPU.

        Args:
            data (cp.array): Input data used for clustering.
            labels (cp.array): Cluster labels assigned to the data points.

        Returns:
            tuple: Silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        silhouette = silhouette_score(data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
        davies_bouldin = davies_bouldin_score(data, cluster_labels)

        score_list = [silhouette, calinski_harabasz, davies_bouldin]
        print("\nClustering Evaluation Metrics:")
        print(f" Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f" Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
        print(f" Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        
        return cluster_labels, score_list
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
    
    def getExecutionMode(self):
        return "cuda"
    
    def setMem(self, path: str, verbose: int = 0):
        """Set memory location and verbosity (not applicable for GPU version)."""
        print('called setMem for CompHelpersGPU: No-op for GPU Version.')
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
    
    def safe_divide(self, a, b):
        return cp.divide(a, b, out=cp.zeros_like(a), where=b!=0)
    
    


    def calc_XZ(self, datasets, args, cache_path, method='weighted', metric='euclidean', norm='ice', overrideCache=False):
        from cupyx.scipy.cluster.hierarchy import linkage as cp_linkage
        import os
        key = tuple(sorted(args)) + (method, metric, norm)
        cache_file = os.path.join(cache_path, f"cache_{key}.pkl")
        
        if not overrideCache:
            try:
                cached_data = cp.load(cache_file + ".npz", allow_pickle=True)
                print(f"Using cached data: {cache_file}.npz")
                return cached_data['X'], cached_data['Z']
            except FileNotFoundError:
                print(f"No cached data, creating cache file {cache_file}")
        else:
            print("Overriding cache, recomputing results")
        
        flat_euclid_dist_map = {}
        max_shape = (0, 0)
        
        for label in args:
            print(f'Processing {label}')
            trajectories = datasets[label]['trajectories']
            if trajectories is None or len(trajectories) == 0:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return cp.array([]), cp.array([])
            
            dist = self.cached_calc_dist(trajectories, metric=metric)
            dist = cp.array(dist)
            print(f"{label} has dist shape {dist.shape}")
            
            # Handle infinite values
            inf_mask = cp.isinf(dist)
            if cp.any(inf_mask):
                print(f"Warning: Infinite values found in distance matrix for {label}. Replacing with large finite value.")
                large_finite = cp.finfo(dist.dtype).max / 2
                dist[inf_mask] = large_finite
            
            # Handle NaN values (likely centromere regions)
            nan_mask = cp.isnan(dist)
            if cp.any(nan_mask):
                print(f"Warning: NaN values found in distance matrix for {label}. These are likely centromere regions.")
                dist[nan_mask] = cp.nanmean(dist)
            
            normalized_dist = cp.array([self.norm_distMatrix(matrix=matrix, norm=norm) for matrix in dist])
            flat_euclid_dist_map[label] = normalized_dist
            
            max_shape = cp.maximum(max_shape, cp.max([d.shape for d in normalized_dist], axis=0))
        
        # Pad arrays to ensure consistent shapes
        padded_flat_euclid_dist_map = {
            label: [cp.pad(val, ((0, max_shape[0] - val.shape[0]), (0, max_shape[1] - val.shape[1]))) 
                    for val in sublist] 
            for label, sublist in flat_euclid_dist_map.items()
        }
        
        # Flatten and stack distance matrices
        flat_euclid_dist_map = {
            label: [padded_flat_euclid_dist_map[label][val][cp.triu_indices_from(padded_flat_euclid_dist_map[label][val], k=1)].flatten()
                    for val in range(len(padded_flat_euclid_dist_map[label]))]
            for label in args
        }
        
        X = cp.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        # Final check for non-finite values
        non_finite_mask = ~cp.isfinite(X)
        if cp.any(non_finite_mask):
            print("Warning: Non-finite values found in flattened distance array. Replacing with mean value.")
            X[non_finite_mask] = cp.nanmean(X)
            
        # Perform linkage
        try:
            Z = cp_linkage(X, method=method, metric='euclidean')
        except ValueError as e:
            print(f"Error in linkage: {e}")
            print("Attempting to proceed with available finite values...")
            finite_mask = cp.isfinite(X)
            X_finite = X[finite_mask]
            Z = cp_linkage(X_finite, method=method, metric='euclidean')
        
        # Cache the results
        if not overrideCache:
            cp.savez_compressed(cache_file, X=X, Z=Z)
        return X, Z