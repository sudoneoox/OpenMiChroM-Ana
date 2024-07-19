from AnalysisTools.Comp_Helper_CPU import ComputeHelpers
from AnalysisTools.Comp_Helper_GPU import ComputeHelpersGPU
from AnalysisTools.Plot_Helper import PlotHelper
from OpenMiChroM.CndbTools import cndbTools

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

import numpy as np
import umap
import ivis
import os 



class Ana:
    def __init__(self, analysisStoragePath: str = "", execution_mode: str = 'cpu', showPlots=True, cacheStoragePath: str=""):
        """__init__

        Initializes the Ana class with a base folder for data storage.

        Args:
            analysisStoragePath (str): directory path to store analysis results (default: {""})
            execution_mode (str): whether to use cpu/cuda for execution (default: {'cpu'})
            showPlots (bool): shows plots from plothelper class instead of having to manually do it (default: {True})
            cacheStoragePath (str): directory path to store cache if (NONE) it wont save to a cache directory (default: {""})
        """  
        
        self.datasets = {}
        self.cndbTools = cndbTools()
        self.execution_mode = execution_mode
        self.showPlots = showPlots
        
      
    
        
        if self.showPlots:
            self.plot_helper = PlotHelper()
        else:
            self.plot_helper = None
        
        if analysisStoragePath == "":
            self.outPath = os.path.join(os.getcwd(), 'Analysis')
            os.makedirs(self.outPath, exist_ok=True)
        else:
            self.outPath = os.path.join(os.getcwd(), analysisStoragePath)
            os.makedirs(self.outPath, exist_ok=True)
            
        if cacheStoragePath == "":
            self.cache_path = os.path.join(os.getcwd(), 'cache')
            os.makedirs(self.cache_path, exist_ok=True)
            self.compute_helpers = ComputeHelpers(memory_location=self.cache_path)
            
        else:
            self.cache_path = os.path.join(os.getcwd(), cacheStoragePath)
            os.makedirs(self.cache_path, exist_ok=True)
            self.compute_helpers = ComputeHelpers(memory_location=self.cache_path)
            self.plot_helper.setMeMForComputeHelpers(cacheStoragePath)
        
        self.execution_mode = execution_mode
        if execution_mode.lower() == "cuda":
            self.compute_helpers = ComputeHelpersGPU()
        elif execution_mode.lower() == "cpu":
            self.compute_helpers = ComputeHelpers()
        else:
            print("Execution mode not valid. Options are cpu/cuda")
            exit()

    def add_dataset(self, label: str, folder: str):
        """add_dataset
        
        Adds a dataset to the analysis.
        
        Args:
            label (str): The label for the dataset.
            folder (str): The folder path containing the dataset.
        """
        self.datasets[label] = {
            'folder': folder,
            'trajectories': None,  # Trajectory data for the dataset
            'distance_array': None,  # distance array
        }


    def process_trajectories(self, label: str, filename: str, folder_pattern: list = ['iteration_', [0, 100]]):
        """
        Processes trajectory data for a given dataset.

        Args:
            label (str): The label for the dataset.
            filename (str): The filename of the trajectory data (.cndb file).
            folder_pattern (list): The folder pattern for iterations.
        """
        config = self.datasets[label]
        trajs_xyz = []
        it_start = folder_pattern[1][0]
        it_end = folder_pattern[1][1]
        inputFolder = os.path.join(config['folder'], folder_pattern[0])

        for i in range(it_start, it_end + 1):
            traj = self.__load_and_process_trajectory(folder=inputFolder, replica=i, filename=filename)
            if traj.size > 0:
                trajs_xyz.append(traj)

        if trajs_xyz:
            max_shape = np.max([traj.shape for traj in trajs_xyz], axis=0)
            trajs_xyz = [self.compute_helpers.pad_array(traj, max_shape) if not np.array_equal(traj.shape, max_shape) else traj for traj in trajs_xyz]
            self.datasets[label]['trajectories'] = np.vstack(trajs_xyz)
            print(f'Trajectory for {label} has shape {self.datasets[label]["trajectories"].shape}')
        else:
            print(f"No valid trajectories found for {label}")

    def __load_and_process_trajectory(self, folder: str, replica: int, filename: str, key: str = None) -> np.array:
        """
        Loads and processes a single trajectory file.

        Args:
            folder (str): The folder containing the trajectory file.
            replica (int): The replica number.
            filename (str): The filename of the trajectory data.
            key (str, optional): Key for bead selection.

        Returns:
            np.array: Processed trajectory data.
        """
        path = f'{folder}{replica}/{filename}'

        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return np.array([])
        else:
            print(f"Processing file: {path}")

        try:
            trajectory = self.cndbTools.load(filename=path)
            list_traj = [int(k) for k in trajectory.cndb.keys() if k != 'types']
            list_traj.sort()
            beadSelection = trajectory.dictChromSeq[key] if key else None
            first_snapshot, last_snapshot = list_traj[0], list_traj[-1]
            trajs_xyz = self.cndbTools.xyz(frames=[first_snapshot, last_snapshot + 1, 2000], XYZ=[0, 1, 2], beadSelection=beadSelection)
            return trajs_xyz
        except Exception as e:
            print(f"Error processing trajectory {replica}: {str(e)}")
            return np.array([])

    """===================================== Analysis ===================================="""

    def dendogram_Z(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice') -> np.array:
        """
        Creates the linkage matrix (dendrogram data) for given labels.

        Args:
            *args: (str): The labels to create the dendrogram for.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.

        Returns:
            np.array: The linkage matrix.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'dendrogram_plot_{args}_{method}_{metric}.png'),
                'title': 'Dendrogram'
            }
        self.plot_helper.plot(plot_type="dendrogram", data=Z, plot_params=plot_params)
        return Z

    def dist_map(self, label: str, method="euclidean") -> np.array:
        """
        Creates the Euclidean distance map for a given dataset.

        Args:
            label (str): Label of the dataset to create the Euclidean distance map for.

        Returns:
            np.array: The Euclidean distance map.
        """
        trajectories = self.datasets[label]['trajectories']

        if trajectories == None:
            print(f"Trajectories not yet loaded for {label}. Load them first")
            return np.array([])

        if self.datasets[label]["distance_array"] == None:
            compute_dist = [cdist(val, val, method) for val in trajectories]
            compute_dist = np.array(compute_dist)
            self.datasets[label]['distance_array'] = compute_dist
            
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'{label}_dist_map.png'),
                'title': f'{label} Distance Map',
                'x_label': 'Beads',
                'y_label': 'Beads'
            }
        self.plot_helper.plot(plot_type="euclidiandistmap", data=[self.datasets[label]["distance_array"]], plot_params=plot_params)

        return self.datasets[label]['distance_array']

    def pca(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', n_components: int = 2) -> tuple:
        """
        Performs PCA on the datasets and returns the principal components and explained variance.

        Args:
            *args: (str): The labels to create the PCA from.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.

        Returns:
            tuple: (np.array, np.array) The principal components and the explained variance ratio.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components = self.compute_helpers.find_optimal_clusters(X, 15)
        pca_result, explained_variance_ratio, components = self.compute_helpers.run_reduction('pca', X, n_components, self.execution_mode, n_jobs=-1)
        if self.showPlots:
            self.plot_helper.plot(plot_type="pcaplot", data=(pca_result, explained_variance_ratio, components), plot_params={
                'outputFileName': os.path.join(self.outPath, f'pca_plot_{args}_{method}_{metric}.png'),
                'title': 'PCA Plot',
                'n_components': n_components
            })
        return pca_result, explained_variance_ratio, components

    def tsne(self, *args: str, tsneParams: dict = None, sample_size: int = 5000, num_clusters: int = -1, method: str = 'weighted', metric: str = "euclidean", norm: str = 'ice') -> tuple:        
        """
        Performs t-SNE on the datasets and returns the t-SNE results and clusters.

        Args:
            *args: (str): The labels to create the t-SNE from.
            tsneParams (dict, optional): Parameters for the t-SNE algorithm. Default is None.
            sample_size (int, optional): The sample size for t-SNE. Default is 5000.
            num_clusters (int, optional): Number of clusters. Default is None.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.

        Returns:
            tuple: (np.array, np.array) The t-SNE results and the clusters.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        tsne_result, kl_divergence, _ = self.compute_helpers.run_reduction('tsne', X, n_components=2, execution_mode=self.execution_mode, **tsneParams or {})
        if self.showPlots:
            self.plot_helper.plot(plot_type="tsneplot", data=(tsne_result, kl_divergence, None), plot_params={
                'outputFileName': os.path.join(self.outPath, f'tsne_plot_{args}_{method}.png'),
                'title': 't-SNE Plot'
            })
        return tsne_result, kl_divergence

    def umap(self, *args: str, umapParams: dict = None, sample_size: int = 5000, num_clusters: int = -1, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice') -> tuple:
        """
        Performs UMAP on the datasets and returns the UMAP results and clusters.

        Args:
            *args: (str): The labels to create the UMAP from.
            umapParams (dict, optional): Parameters for the UMAP algorithm. Default is None.
            sample_size (int, optional): The sample size for UMAP. Default is 5000.
            num_clusters (int, optional): Number of clusters. Default is None.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.

        Returns:
            tuple: (np.array, np.array) The UMAP results and the clusters.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        umap_result = self.compute_helpers.run_reduction('umap', X, n_components=2, execution_mode=self.execution_mode, **umapParams or {})
        if self.showPlots:
            self.plot_helper.plot(plot_type="umapplot", data=umap_result, plot_params={
                'outputFileName': os.path.join(self.outPath, f'umap_plot_{args}_{method}.png'),
                'title': 'UMAP Plot'
            })
        return umap_result


        return umap_res, fclust

    def ivis_clustering(self, *args: str, ivisParams: dict = None, sample_size: int = 5000, num_clusters: int = -1, method: str = 'weighted', metric: str="euclidean", norm: str = 'ice') -> tuple:
        """
        Performs IVIS on the datasets and returns the IVIS results and clusters.

        Args:
            *args: (str): The labels to create the IVIS from.
            ivisParams (dict, optional): Parameters for the IVIS algorithm. Default is None.
            sample_size (int, optional): The sample size for IVIS. Default is 5000.
            num_clusters (int, optional): Number of clusters. Default is None.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): the metric for hierarchical clustering. Default is euclidean

        Returns:
            tuple: (np.array, np.array) The IVIS results and the clusters.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        ivis_result = self.compute_helpers.run_reduction('ivis', X, n_components=2, execution_mode=self.execution_mode, **ivisParams or {})
        if self.showPlots:
            self.plot_helper.plot(plot_type="ivisplot", data=ivis_result, plot_params={
                'outputFileName': os.path.join(self.outPath, f'ivis_plot_{args}_{method}.png'),
                'title': 'IVIS Plot'
            })
        return ivis_result
    
    
    
    def svd(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', n_components: int = 2) -> tuple:
        """
        Performs Singular Value Decomposition (SVD) on the datasets.

        Args:
            *args (str): The labels to create the SVD from.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.
            norm (str, optional): The normalization method to use. Default is 'ice'.
            n_components (int, optional): Number of components to keep. Default is 2.

        Returns:
            tuple: (np.array, np.array, np.array) The SVD results, singular values, and right singular vectors.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components == self.compute_helpers.find_optimal_clusters(X)
        svd_result, singular_values, vt = self.compute_helpers.run_reduction('svd', X, n_components, self.execution_mode, n_jobs=-1)
        if self.showPlots:
            self.plot_helper.plot(plot_type="svdplot", data=(svd_result, singular_values, vt), plot_params={
                'outputFileName': os.path.join(self.outPath, f'svd_plot_{args}_{method}_{metric}.png'),
                'title': 'SVD Plot',
                'n_components': n_components
            })
        return svd_result, singular_values, vt

    def mds(self, *args: str, mdsParams: dict = None, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', n_components: int = 2) -> tuple:
        """
        Performs Multidimensional Scaling (MDS) on the datasets.

        Args:
            *args (str): The labels to create the MDS from.
            mdsParams (dict, optional): Parameters for the MDS algorithm. Default is None.
            sample_size (int, optional): The sample size for MDS. Default is 5000.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.
            norm (str, optional): The normalization method to use. Default is 'ice'.
            n_components (int, optional): Number of components to keep. Default is 2.

        Returns:
            tuple: (np.array, float, np.array) The MDS results, stress value, and dissimilarity matrix.
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components = self.compute_helpers.find_optimal_clusters(X)
            
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        mds_result, stress, dissimilarity_matrix = self.compute_helpers.run_reduction('mds', X, n_components, self.execution_mode, **mdsParams or {})
        if self.showPlots:
            self.plot_helper.plot(plot_type="mdsplot", data=(mds_result, stress, dissimilarity_matrix), plot_params={
                'outputFileName': os.path.join(self.outPath, f'mds_plot_{args}_{method}_{metric}.png'),
                'title': 'MDS Plot',
                'n_components': n_components
            })
        return mds_result, stress, dissimilarity_matrix
    
    def spectral_clustering(self, *args: str, spectralParams: dict = None, sample_size: int = 5000, num_clusters: int = -1, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice') -> tuple:
        """
        Performs spectral clustering on the datasets and returns the clustering results.

        Args:
            *args: (str): The labels to create the spectral clustering from.
            spectralParams (dict, optional): Parameters for the spectral clustering algorithm.
            sample_size (int, optional): The sample size for spectral clustering. Default is 5000.
            num_clusters (int, optional): Number of clusters. If None, it will be determined automatically.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use. Default is 'euclidean'.

        Returns:
            tuple: (cluster_labels, eigenvalues, eigenvectors, affinity_matrix, silhouette_avg, calinski_harabasz, davies_bouldin)
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        spectral_result = self.compute_helpers.run_clustering('spectral', X, execution_mode=self.execution_mode, n_clusters=num_clusters, **spectralParams or {})
        if self.showPlots:
            self.plot_helper.plot(plot_type="spectralclusteringplot", data=[X, spectral_result], plot_params={
                'outputFileName': os.path.join(self.outPath, f'spectral_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'Spectral Clustering Plot'
            })
        return spectral_result
    
    
    def kmeans_clustering(self, *args: str, n_clusters: int = 5, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', **kwargs) -> tuple:
        """
        Performs K-means clustering on the datasets.

        Args:
            *args: (str): The labels to perform clustering on.
            n_clusters (int): Number of clusters. Default is 5.
            sample_size (int): The sample size for clustering. Default is 5000.
            method (str): The method for hierarchical clustering. Default is 'weighted'.
            metric (str): The distance metric to use. Default is 'euclidean'.
            norm (str): The normalization method. Default is 'ice'.
            **kwargs: Additional arguments for KMeans.

        Returns:
            tuple: (cluster_labels, cluster_centers)
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        kmeans_result = self.compute_helpers.run_clustering('kmeans', X, execution_mode=self.execution_mode, n_clusters=n_clusters, **kwargs)
        
        if self.showPlots:
            self.plot_helper.plot(plot_type="kmeans", data=[X, kmeans_result[0]], plot_params={
                'outputFileName': os.path.join(self.outPath, f'kmeans_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'K-means Clustering Plot'
            })
        
        return kmeans_result

    def dbscan_clustering(self, *args: str, eps: float = 0.5, min_samples: int = 5, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', **kwargs) -> np.array:
        """
        Performs DBSCAN clustering on the datasets.

        Args:
            *args: (str): The labels to perform clustering on.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default is 0.5.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is 5.
            sample_size (int): The sample size for clustering. Default is 5000.
            method (str): The method for hierarchical clustering. Default is 'weighted'.
            metric (str): The distance metric to use. Default is 'euclidean'.
            norm (str): The normalization method. Default is 'ice'.
            **kwargs: Additional arguments for DBSCAN.

        Returns:
            np.array: Cluster labels
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        dbscan_result = self.compute_helpers.run_clustering('dbscan', X, execution_mode=self.execution_mode, eps=eps, min_samples=min_samples, **kwargs)
        
        if self.showPlots:
            self.plot_helper.plot(plot_type="dbscan", data=[X, dbscan_result], plot_params={
                'outputFileName': os.path.join(self.outPath, f'dbscan_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'DBSCAN Clustering Plot'
            })
        
        return dbscan_result

    def hierarchical_clustering(self, *args: str, n_clusters: int = 5, sample_size: int = 5000, method: str = 'ward', metric: str = 'euclidean', norm: str = 'ice', **kwargs) -> np.array:
        """
        Performs hierarchical clustering on the datasets.

        Args:
            *args: (str): The labels to perform clustering on.
            n_clusters (int): Number of clusters. Default is 5.
            sample_size (int): The sample size for clustering. Default is 5000.
            method (str): The linkage method to use. Default is 'ward'.
            metric (str): The distance metric to use. Default is 'euclidean'.
            norm (str): The normalization method. Default is 'ice'.
            **kwargs: Additional arguments for hierarchical clustering.

        Returns:
            np.array: Cluster labels
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        hierarchical_result = self.compute_helpers.run_clustering('hierarchical', X, execution_mode=self.execution_mode, n_clusters=n_clusters, linkage_method=method, **kwargs)
        
        if self.showPlots:
            self.plot_helper.plot(plot_type="hierarchical", data=[X, hierarchical_result], plot_params={
                'outputFileName': os.path.join(self.outPath, f'hierarchical_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'Hierarchical Clustering Plot'
            })
        
        return hierarchical_result

    def optics_clustering(self, *args: str, min_samples: int = 5, xi: float = 0.05, min_cluster_size: float = 0.05, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', **kwargs) -> np.array:
        """
        Performs OPTICS clustering on the datasets.

        Args:
            *args: (str): The labels to perform clustering on.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is 5.
            xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. Default is 0.05.
            min_cluster_size (float): Minimum number of samples in an OPTICS cluster. Default is 0.05.
            sample_size (int): The sample size for clustering. Default is 5000.
            method (str): The method for hierarchical clustering. Default is 'weighted'.
            metric (str): The distance metric to use. Default is 'euclidean'.
            norm (str): The normalization method. Default is 'ice'.
            **kwargs: Additional arguments for OPTICS.

        Returns:
            np.array: Cluster labels
        """
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        optics_result = self.compute_helpers.run_clustering('optics', X, execution_mode=self.execution_mode, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, **kwargs)
        
        if self.showPlots:
            self.plot_helper.plot(plot_type="optics", data=[X, optics_result], plot_params={
                'outputFileName': os.path.join(self.outPath, f'optics_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'OPTICS Clustering Plot'
            })
        
        return optics_result


    """=========================================UTILITIES========================================"""

    def calc_XZ(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', n_jobs: int = -1, norm: str = 'ice') -> tuple:
        key = tuple(sorted(args)) + (method, metric, norm)
        cache_file = os.path.join(self.cache_path, f"cache_{key}.pkl")
        try:
            cached_data = np.load(cache_file + ".npz", allow_pickle=True)
            print(f"using cached data: {cache_file}.npz")
            return cached_data['X'], cached_data['Z']
        except FileNotFoundError:
            print(f"No cached data, creating cache file {cache_file}")
        
        flat_euclid_dist_map = {}
        max_shape = (0, 0)
        
        for label in args:
            print(f'Processing {label}')
            trajectories = self.datasets[label]['trajectories']
            if trajectories is None or len(trajectories) == 0:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return np.array([]), np.array([])
            
            dist = self.compute_helpers.cached_calc_dist(trajectories, metric, self.execution_mode, n_jobs)
            dist = np.array(dist)
            print(f"{label} has dist shape {dist.shape}")
            
            normalized_dist = np.array([self.compute_helpers.norm_distMatrix(matrix=matrix, norm=norm) for matrix in dist])
            flat_euclid_dist_map[label] = normalized_dist
            
            max_shape = np.maximum(max_shape, np.max([d.shape for d in normalized_dist], axis=0))
        
        padded_flat_euclid_dist_map = {label: [np.pad(val, ((0, max_shape[0] - val.shape[0]), (0, max_shape[1] - val.shape[1]))) for val in sublist] for label, sublist in flat_euclid_dist_map.items()}
        
        flat_euclid_dist_map = {label: [padded_flat_euclid_dist_map[label][val][np.triu_indices_from(padded_flat_euclid_dist_map[label][val], k=1)].flatten()
                                        for val in range(len(padded_flat_euclid_dist_map[label]))]
                                for label in args}
        
        X = np.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        # Z = self.compute_helpers.hierarchical_clustering(X, method=method)
        Z = linkage(X, method=method, metric='euclidean')
        
        np.savez_compressed(cache_file, X=X, Z=Z)
        return X, Z
        
        
    
    
    """ ============================================================= Getters/Setters ============================================================================================"""

    def getCachePath(self):
        return self.cache_path
    
    def getAnalysisPath(self):
        return self.outPath
    
    def getExecutionMode(self):
        return self.execution_mode
    
    def getShowPlots(self):
        return self.showPlots
    
    def setAnalysisPath(self, path: str):
        self.outPath = os.path.join(os.getcwd(), path)
        os.makedirs(self.outPath, exist_ok=True)
    
    def setCachePath(self, path: str):
        self.cache_path = os.path.join(os.getcwd(), path)
        os.makedirs(self.cache_path, exist_ok=True)
    
    def setExecutionMode(self, execMode: str):
        self.execution_mode = execMode
    
    def setShowPlots(self, show: bool):
        self.showPlots = show