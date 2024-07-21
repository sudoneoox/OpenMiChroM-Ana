import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed, Memory
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
import umap.umap_ as umap
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import os

@jit(nopython=True)
def _ice_normalization_numba(matrix, max_iter=100, tolerance=1e-5):
    """
    Perform ICE normalization using Numba for improved performance.
    
    Args:
        matrix (np.array): Input matrix to normalize.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
    
    Returns:
        np.array: Normalized matrix.
    """
    n = matrix.shape[0]
    bias = np.ones(n)
    matrix_balanced = matrix.copy()
    
    for _ in range(max_iter):
        bias_old = bias.copy()
        row_sums = matrix_balanced.sum(axis=1)
        col_sums = matrix_balanced.sum(axis=0)
        
        for i in range(n):
            if row_sums[i] != 0 and col_sums[i] != 0:
                bias[i] *= np.sqrt(row_sums[i] * col_sums[i])
            else:
                bias[i] = 1
        
        for i in range(n):
            for j in range(n):
                if bias[i] != 0 and bias[j] != 0:
                    matrix_balanced[i, j] = matrix[i, j] / (bias[i] * bias[j])
                else:
                    matrix_balanced[i, j] = 0
        
        if np.sum(np.abs(bias - bias_old)) < tolerance:
            break
    
    return matrix_balanced

@jit(nopython=True)
def _kr_norm_numba(matrix, max_iter=100, tolerance=1e-5):
    """
    Perform KR normalization using Numba for improved performance.
    
    Args:
        matrix (np.array): Input matrix to normalize.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
    
    Returns:
        np.array: Normalized matrix.
    """
    n = matrix.shape[0]
    bias = np.ones(n)
    matrix_balanced = matrix.copy()
    
    for _ in range(max_iter):
        bias_old = bias.copy()
        row_sums = matrix_balanced.sum(axis=1)
        col_sums = matrix_balanced.sum(axis=0)
        
        for i in range(n):
            if bias[i] != 0:
                bias[i] *= np.sqrt(row_sums[i] * col_sums[i])
            else:
                bias[i] = 1
        
        for i in range(n):
            for j in range(n):
                matrix_balanced[i, j] = matrix[i, j] / (bias[i] * bias[j])
        
        if np.sum(np.abs(bias - bias_old)) < tolerance:
            break
     
    return matrix_balanced

class ComputeHelpers:
    """
    A class containing helper functions for various computational tasks related to HiC data analysis.
    This version supports multithreading for improved performance on CPU.
    """

    def __init__(self, memory_location: str = '.', memory_verbosity: int = 0):
        """
        Initialize the ComputeHelpers class.

        Args:
            memory_location (str): The location for caching computations.
            memory_verbosity (int): The verbosity level for memory caching.
        """
        self.memory = Memory(location=memory_location, verbose=memory_verbosity)
        self.n_jobs = os.cpu_count()  # Default to using all available cores
        
        self.reduction_methods = {
            'pca': self._pca_reduction,
            'svd': self._svd_reduction,
            'tsne': self._tsne_reduction,
            'umap': self._umap_reduction,
            'mds': self._mds_reduction
        }
        
        self.clustering_methods = {
            'dbscan': self._dbscan_clustering,
            'optics': self._optics_clustering,
            'hierarchical': self._hierarchical_clustering,
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

    def set_n_jobs(self, n_jobs: int):
        """
        Set the number of jobs for parallel processing.

        Args:
            n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        """
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

    '''#!========================================================== DATA LOADING AND PREPROCESSING ====================================================================================='''

    def getHiCData_simulation(self, filepath):
        """
        Load and process simulated HiC data.

        Args:
            filepath (str): Path to the input file.

        Returns:
            tuple: Processed HiC matrix, scaling vector, and error vector.
        """
        contactMap = np.loadtxt(filepath)
        r = np.triu(contactMap, k=1) 
        r = normalize(r, axis=1, norm='max') 
        rd = np.transpose(r) 
        r = r + rd + np.diag(np.ones(len(r))) 

        D1 = []
        err = []
        for i in range(0, np.shape(r)[0]): 
            D1.append((np.mean(np.diag(r,k=i)))) 
            err.append((np.std(np.diag(r,k=i))))
        
        return r, D1, err

    def getHiCData_experiment(self, filepath, cutoff=0.0, norm="max"):
        """
        Load and process experimental HiC data.

        Args:
            filepath (str): Path to the input file.
            cutoff (float): Cutoff value for filtering data.
            norm (str): Normalization method.

        Returns:
            tuple: Processed HiC matrix, scaling vector, and error vector.
        """
        contactMap = np.loadtxt(filepath)
        r = np.triu(contactMap, k=1)
        r[np.isnan(r)] = 0.0
        r = normalize(r, axis=1, norm="max")
        
        if norm == "first":
            for i in range(len(r) - 1):
                maxElem = r[i][i + 1]
                if(maxElem != np.max(r[i])):
                    for j in range(len(r[i])):
                        if maxElem != 0.0:
                            r[i][j] = float(r[i][j] / maxElem)
                        else:
                            r[i][j] = 0.0 
                        if r[i][j] > 1.0:
                            r[i][j] = .5
        r[r<cutoff] = 0.0
        rd = np.transpose(r) 
        r = r + rd + np.diag(np.ones(len(r)))

        D1 = []
        err = []
        for i in range(0, np.shape(r)[0]): 
            D1.append((np.mean(np.diag(r,k=i)))) 
            err.append((np.std(np.diag(r,k=i))))
        D = np.array(D1)
        err = np.array(err)

        return r, D, err

    def pad_array(self, array, target_shape):
        """
        Pad a 3D array to the target shape with zeros.

        Args:
            array (np.array): The array to pad.
            target_shape (tuple): The target shape to pad to.

        Returns:
            np.array: The padded array.
        """
        result = np.zeros(target_shape)
        slices = tuple(slice(0, dim) for dim in array.shape)
        result[slices] = array
        return result
    
    '''#!========================================================== DISTANCE CALCULATIONS ====================================================================================='''

    def calc_dist(self, X: np.array, metric: str) -> np.array:
        """
        Calculate the distance matrix using the specified metric.

        Args:
            X (np.array): The input data matrix.
            metric (str): The distance metric to use.

        Returns:
            np.array: The calculated distance matrix.
        """
        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                return executor.submit(self.distance_metrics[metric], X).result()
        except KeyError:
            raise KeyError(f"Invalid metric: {metric}. Available metrics are: {list(self.distance_metrics.keys())}")

    def euclidean_distance(self, x):
        """
        Calculate the Euclidean distance matrix.

        Args:
            x (np.array): Input data.

        Returns:
            np.array: Euclidean distance matrix.
        """
        return squareform(pdist(x, metric='euclidean'))

    def contact_distance(self, x):
        """
        Calculate the contact distance matrix.

        Args:
            x (np.array): Input data.

        Returns:
            np.array: Contact distance matrix.
        """
        return squareform(pdist(x, metric='cityblock'))

    def pearson_distance(self, X: np.array) -> np.array:
        """
        Calculate Pearson correlation distance.

        Args:
            X (np.array): The input data matrix.

        Returns:
            np.array: The Pearson correlation distance matrix.
        """
        corr = np.corrcoef(X)
        return 1 - corr

    def spearman_distance(self, X: np.array) -> np.array:
        """
        Calculate Spearman correlation distance.

        Args:
            X (np.array): The input data matrix.

        Returns:
            np.array: The Spearman correlation distance matrix.
        """
        rank_data = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, X)
        return 1 - np.corrcoef(rank_data)

    def log2_contact_distance(self, X: np.array) -> np.array:
        """
        Calculate log2 contact distance.

        Args:
            X (np.array): The input data matrix.

        Returns:
            np.array: The log2 contact distance matrix.
        """
        log2_X = np.log2(X + 1)
        return squareform(pdist(log2_X, metric='cityblock'))
    
    '''#!========================================================== NORMALIZATION METHODS ====================================================================================='''

    def norm_distMatrix(self, matrix: np.array, norm: str):
        """
        Normalize the distance matrix using the specified method.

        Args:
            matrix (np.array): The input distance matrix.
            norm (str): The normalization method to use.

        Returns:
            np.array: The normalized distance matrix.
        """        
        try:
            return self.normalization_methods[norm](matrix)
        except KeyError:
            raise KeyError(f"Invalid normalization method: {norm}. Available methods are: {list(self.normalization_methods.keys())}")

    def log_transform(self, m):
        """
        Perform log transformation on the input matrix.

        Args:
            m (np.array): Input matrix.

        Returns:
            np.array: Log-transformed matrix.
        """
        return np.log2(m + 1)

    def vc_normalization(self, m):
        """
        Perform variance stabilizing normalization on the input matrix.

        Args:
            m (np.array): Input matrix.

        Returns:
            np.array: Normalized matrix.
        """
        return m / m.sum(axis=1, keepdims=True)

    def ice_normalization(self, matrix: np.array, max_iter: int=100, tolerance: float=1e-5) -> np.array:
        """
        Perform ICE normalization on the input matrix.

        Args:
            matrix (np.array): Input matrix.
            max_iter (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.

        Returns:
            np.array: ICE normalized matrix.
        """
        return _ice_normalization_numba(matrix, max_iter, tolerance)

    def kr_norm(self, matrix: np.array, max_iter: int=100, tolerance: float=1e-5) -> np.array:
        """
        Perform KR normalization on a contact matrix.

        Args:
            matrix (np.array): Raw contact matrix.
            max_iter (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.

        Returns:
            np.array: Normalized contact matrix.
        """
        return _kr_norm_numba(matrix, max_iter, tolerance)
    
    '''#!========================================================== DIMENSIONALITY REDUCTION METHODS ====================================================================================='''

    def run_reduction(self, method, X, n_components):
        """
        Run the specified dimensionality reduction method.

        Args:
            method (str): The reduction method to use.
            X (np.array): Input data.
            n_components (int): Number of components for the reduction.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            tuple: Reduced data and additional information (if available).
        """
        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                return executor.submit(self.reduction_methods[method], X, n_components, self.n_jobs).result()
        except KeyError:
            raise KeyError(f"Invalid reduction method: {method}. Available methods are: {list(self.reduction_methods.keys())}")

    def _pca_reduction(self, X, n_components):
        """
        Perform PCA reduction.

        Args:
            X (np.array): Input data.
            n_components (int): Number of components.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            tuple: PCA result, explained variance ratio, and components.
        """
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(X)
        return result, pca.explained_variance_ratio_, pca.components_

    def _svd_reduction(self, X, n_components):
        """
        Perform SVD reduction.

        Args:
            X (np.array): Input data.
            n_components (int): Number of components.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            np.array: SVD result.
        """
        svd = TruncatedSVD(n_components=n_components)
        result = svd.fit_transform(X)
        return result

    def _tsne_reduction(self, X, n_components):
            """
            Perform t-SNE reduction.

            Args:
                X (np.array): Input data.
                n_components (int): Number of components.
                n_jobs (int): Number of jobs to run in parallel.

            Returns:
                tuple: t-SNE result, KL divergence, and None (for consistency with other methods).
            """
            tsne = TSNE(n_components=n_components, n_jobs=self.n_jobs)
            result = tsne.fit_transform(X)
            return result, tsne.kl_divergence_, None

    def _umap_reduction(self, X, n_components):
        """
        Perform UMAP reduction.

        Args:
            X (np.array): Input data.
            n_components (int): Number of components.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            tuple: UMAP result, embedding, and graph.
        """
        umap_reducer = umap.UMAP(n_components=n_components, n_jobs=self.n_jobs)
        result = umap_reducer.fit_transform(X)
        return result, umap_reducer.embedding_, umap_reducer.graph_

    def _mds_reduction(self, X, n_components):
        """
        Perform MDS reduction.

        Args:
            X (np.array): Input data.
            n_components (int): Number of components.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            tuple: MDS result, stress, and dissimilarity matrix.
        """
        mds = MDS(n_components=n_components, n_jobs=self.n_jobs)
        result = mds.fit_transform(X)
        return result, mds.stress_, mds.dissimilarity_matrix_

    '''#!========================================================== CLUSTERING METHODS ====================================================================================='''

    def run_clustering(self, method, X, **kwargs):
        """
        Run the specified clustering method.

        Args:
            method (str): The clustering method to use.
            X (np.array): Input data.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for the clustering method.

        Returns:
            np.array: Cluster labels.
        """
        try:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                return executor.submit(self.clustering_methods[method], X, n_jobs=self.n_jobs, **kwargs).result()
        except KeyError:
            raise KeyError(f"Invalid clustering method: {method}. Available methods are: {list(self.clustering_methods.keys())}")

    def _kmeans_clustering(self, X: np.array, n_clusters: int, **kwargs):
        """
        Perform K-means clustering.

        Args:
            X (np.array): Input data.
            n_clusters (int): Number of clusters.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for KMeans.

        Returns:
            np.array: Cluster labels.
        """
        return KMeans(n_clusters=n_clusters, n_jobs=self.n_jobs, **kwargs).fit_predict(X)

    def _spectral_clustering(self, X, n_clusters, **kwargs):
        """
        Perform Spectral clustering.

        Args:
            X (np.array): Input data.
            n_clusters (int): Number of clusters.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for SpectralClustering.

        Returns:
            np.array: Cluster labels.
        """
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=n_clusters, n_jobs=self.n_jobs, **kwargs).fit_predict(X)

    def _hierarchical_clustering(self, X, n_clusters, **kwargs):
        """
        Perform Hierarchical clustering.

        Args:
            X (np.array): Input data.
            n_clusters (int): Number of clusters.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for linkage and fcluster.

        Returns:
            np.array: Cluster labels.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(X, method=kwargs.get('method', 'ward'), metric=kwargs.get('metric', 'euclidean'))
        return fcluster(Z, t=n_clusters, criterion='maxclust')

    def _dbscan_clustering(self, X, eps, min_samples, **kwargs):
        """
        Perform DBSCAN clustering.

        Args:
            X (np.array): Input data.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for DBSCAN.

        Returns:
            np.array: Cluster labels.
        """
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs, **kwargs).fit_predict(X)

    def _optics_clustering(self, X, min_samples, xi, min_cluster_size, **kwargs):
        """
        Perform OPTICS clustering.

        Args:
            X (np.array): Input data.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
            min_cluster_size (int): Minimum number of samples in an OPTICS cluster.
            n_jobs (int): Number of jobs to run in parallel.
            **kwargs: Additional arguments for OPTICS.

        Returns:
            np.array: Cluster labels.
        """
        from sklearn.cluster import OPTICS
        return OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, n_jobs=self.n_jobs, **kwargs).fit_predict(X)

    def find_optimal_clusters(self, data: np.array, max_clusters: int=10):
        """
        Find the optimal number of clusters using the elbow method and silhouette score.

        Args:
            data (np.array): Input data for clustering.
            max_clusters (int): Maximum number of clusters to consider.

        Returns:
            int: Optimal number of clusters.
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        
        # Use the elbow method to find the optimal number of clusters
        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        elbow = kl.elbow if kl.elbow else max_clusters
        
        # Find the number of clusters with the highest silhouette score
        silhouette_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Return the smaller of the two to be conservative
        return min(elbow, silhouette_optimal)

    def evaluate_clustering(self, data, labels):
        """
        Evaluate clustering quality using various metrics.

        Args:
            data (np.array): Input data used for clustering.
            labels (np.array): Cluster labels assigned to the data points.

        Returns:
            tuple: Silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
        """
        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        print("\nClustering Evaluation Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        
        return silhouette, calinski_harabasz, davies_bouldin

    """==================================================================== UTILITY METHODS =========================================================="""

    def _calc_dist_wrapper(self, trajectories, metric):
        """
        Wrapper function for calc_dist to be used with joblib caching.

        Args:
            trajectories (list): List of trajectory data.
            metric (str): Distance metric to use.
            n_jobs (int): Number of jobs for parallel computation.

        Returns:
            list: List of distance matrices.
        """
        return Parallel(n_jobs=self.n_jobs)(delayed(self.calc_dist)(val, metric) for val in trajectories)

    def cached_calc_dist(self, trajectories, metric, n_jobs):
        """
        Caches the calculation of distance matrices.

        Args:
            trajectories (list): List of trajectory data.
            metric (str): Distance metric to use.
            n_jobs (int): Number of jobs for parallel computation.

        Returns:
            list: List of cached distance matrices.
        """
        return self.memory.cache(self._calc_dist_wrapper)(trajectories, metric, n_jobs)

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
        """Get memory statistics."""
        return [self.memory.location, self.memory.verbose]
    
    def setMem(self, path: str, verbose: int = 0):
        """Set memory location and verbosity."""
        self.memory = Memory(location=path, verbose=verbose)