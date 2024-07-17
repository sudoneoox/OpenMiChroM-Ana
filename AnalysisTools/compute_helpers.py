import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from numba import jit, cuda
from joblib import Parallel, delayed, Memory
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from kneed import KneeLocator
from sklearn.cluster import KMeans

class ComputeHelpers:
    """
    A class containing helper functions for various computational tasks related to HiC data analysis.
    """

    def __init__(self, memory_location: str = '.', memory_verbosity: int = 0):
        """
        Initialize the ComputeHelpers class.

        Args:
            memory_location (str): The location for caching computations.
            memory_verbosity (int): The verbosity level for memory caching.
        """
        self.memory = Memory(location=memory_location, verbose=memory_verbosity)
        self.CUDA_AVAILABLE = False
        self.cached_calc_dist = self.memory.cache(self._calc_dist_wrapper)

    def set_cuda_availability(self, available):
        """
        Set the availability of CUDA for GPU computations.

        Args:
            available (bool): Whether CUDA is available.
        """
        self.CUDA_AVAILABLE = available
        if self.CUDA_AVAILABLE:
            import cupy as cp

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
        
        # Apply first normalization if specified
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

    def calc_dist(self, X: np.array, metric: str) -> np.array:
        """
        Calculate the distance matrix using the specified metric.

        Args:
            X (np.array): The input data matrix.
            metric (str): The distance metric to use.

        Returns:
            np.array: The calculated distance matrix.
        """
        metricPtrs = {
            'euclidean': lambda x: squareform(pdist(x, metric='euclidean')),
            'pearsons': self.pearson_distance,
            'spearman': self.spearman_distance,
            'contact': lambda x: squareform(pdist(x, metric='cityblock')),
            "log2_contact": self.log2_contact_distance
        }
        try:
            return metricPtrs[metric](X)
        except KeyError:
            raise KeyError(f"Invalid metric: {metric}. Available metrics are: {list(metricPtrs.keys())}")

    @cuda.jit
    def gpu_distance(self, traj, dist, metric):
        """
        CUDA kernel for calculating distances on GPU.

        Args:
            traj (np.array): The trajectory data.
            dist (np.array): The output distance matrix.
            metric (str): The distance metric to use.
        """
        i, j = cuda.grid(2)
        if i < traj.shape[0] and j < traj.shape[0]:
            if metric == 'euclidean':
                dist[i, j] = self.gpu_euclidean_distance(traj[i], traj[j])
            elif metric == 'pearson':
                dist[i, j] = self.gpu_pearson_distance(traj[i], traj[j])
            elif metric == 'spearman':
                dist[i, j] = self.gpu_spearman_distance(traj[i], traj[j])
            elif metric == 'contact':
                dist[i, j] = self.gpu_contact_distance(traj[i], traj[j])
            elif metric == 'log2_contact':
                dist[i, j] = self.gpu_log2_contact_distance(traj[i], traj[j])

    def calc_distance_gpu(self, traj, metric):
        """
        Calculate distance matrix using GPU.

        Args:
            traj (np.array): The trajectory data.
            metric (str): The distance metric to use.

        Returns:
            np.array: The calculated distance matrix.
        """
        traj = np.array(traj)
        dist = np.zeros((traj.shape[0], traj.shape[0]), dtype=np.float32)
        threadsperblock = (16, 16)
        blockspergrid_x = (traj.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (traj.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        self.gpu_distance[blockspergrid, threadsperblock](traj, dist, metric)
        return dist

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

    def norm_distMatrix(self, matrix: np.array, norm: str):
        """
        Normalize the distance matrix using the specified method.

        Args:
            matrix (np.array): The input distance matrix.
            norm (str): The normalization method to use.

        Returns:
            np.array: The normalized distance matrix.
        """
        normPtrs = {
            'ice': self.ice_normalization,
            'log_transform': lambda m: np.log2(m + 1),
            'vc': lambda m: m / m.sum(axis=1, keepdims=True),
            'kr': self.kr_norm,
        }
        try:
            return normPtrs[norm](matrix)
        except KeyError:
            raise KeyError(f"Invalid normalization method: {norm}. Available methods are: {list(normPtrs.keys())}")

    # GPU-specific distance calculation methods
    @cuda.jit(device=True)
    def gpu_euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2))

    @cuda.jit(device=True)
    def gpu_pearson_distance(self, a, b):
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        std_a = np.sqrt(np.sum((a - mean_a)**2) / (len(a) - 1))
        std_b = np.sqrt(np.sum((b - mean_b)**2) / (len(b) - 1))
        cov = np.sum((a - mean_a) * (b - mean_b)) / (len(a) - 1)
        return 1 - (cov / (std_a * std_b))

    @cuda.jit(device=True)
    def gpu_spearman_distance(self, a, b):
        def rank(x):
            temp = sorted(enumerate(x), key=lambda x: x[1])
            ranks = [0] * len(x)
            for i, (index, value) in enumerate(temp):
                ranks[index] = i
            return ranks
        
        rank_a = rank(a)
        rank_b = rank(b)
        return self.gpu_pearson_distance(rank_a, rank_b)

    @cuda.jit(device=True)
    def gpu_contact_distance(self, a, b):
        return np.sum(np.abs(a - b))

    @cuda.jit(device=True)
    def gpu_log2_contact_distance(self, a, b):
        log2_a = np.log2(a + 1)
        log2_b = np.log2(b + 1)
        return np.sum(np.abs(log2_a - log2_b))
    
    #CPU Multithreaded - specific norm calculations

    @jit(nopython=True)
    def ice_normalization(self, matrix: np.array, max_iter: int=100, tolerance: float=1e-5) -> np.array:
        """
        Perform ICE normalization on a contact matrix.

        Args:
            matrix (np.array): Raw contact matrix.
            max_iter (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.

        Returns:
            np.array: Normalized contact matrix.
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

    @jit(nopython=True)
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

    def _calc_dist_wrapper(self, trajectories, metric, execution_mode, n_jobs):
        """
        Wrapper function for calc_dist to be used with joblib caching.

        Args:
            trajectories (list): List of trajectory data.
            metric (str): Distance metric to use.
            execution_mode (str): 'cuda' for GPU, otherwise CPU.
            n_jobs (int): Number of jobs for parallel CPU computation.

        Returns:
            list: List of distance matrices.
        """
        if execution_mode == 'cuda' and self.CUDA_AVAILABLE:
            return [self.calc_distance_gpu(traj, metric) for traj in trajectories]
        else:
            return Parallel(n_jobs=n_jobs)(delayed(self.calc_dist)(val, metric) for val in trajectories)

    def cached_calc_dist(self, trajectories, metric, execution_mode, n_jobs):
        """
        Calculate distance matrix with caching and potential GPU acceleration.

        This method is now defined in __init__ using self.memory.cache
        """
        pass  # The actual implementation is now in _calc_dist_wrapper


    def find_optimal_clusters(self, data, max_clusters=10):
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
    
    
    
    """==================================================================== SETTERS/GETTERS =========================================================="""
    
    def setMem(self, path: str, verbose: int = 0):
        self.memory = Memory(location=path, verbose=verbose)
 