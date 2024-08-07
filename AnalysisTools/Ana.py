from AnalysisTools.PlotHelper import PlotHelper
from OpenMiChroM.CndbTools import cndbTools

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

import numpy as np
import os 



class Ana:
    def __init__(
        self,  
        execution_mode: str = 'cpu', 
        analysisStoragePath: str = "",
        cacheStoragePath: str="",
        showPlots=True, 
        **kwargs
        ):
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
        
        if analysisStoragePath == "":
            self.outPath = os.path.join(os.getcwd(), 'Analysis')
            os.makedirs(self.outPath, exist_ok=True)
        else:
            self.outPath = os.path.join(os.getcwd(), analysisStoragePath)
            os.makedirs(self.outPath, exist_ok=True)
            
        self.setExecutionMode(execution_mode, **kwargs)
        
        if self.showPlots:
            self.plot_helper = PlotHelper()
        else:
            self.plot_helper = None
            
        
        if cacheStoragePath == "":
            self.cache_path = os.path.join(os.getcwd(), 'cache')
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            self.cache_path = os.path.join(os.getcwd(), cacheStoragePath)
            os.makedirs(self.cache_path, exist_ok=True)
            self.plot_helper.setMeMForComputeHelpers(memory_location=cacheStoragePath)
        self.compute_helpers.setMem(memory_location=self.cache_path)

            
            
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


    def process_trajectories(self, label: str, filename: str, folder_pattern: list = ['iteration_', [0, 100]], cache_trajs: bool = False):
        """
        Processes trajectory data for a given dataset.

        Args:
            label (str): The label for the dataset.
            filename (str): The filename of the trajectory data (.cndb file).
            folder_pattern (list): The folder pattern for iterations.
            cache_trajs (bool): saves the loaded cache_trajs in self.cache_path/{label}_trajs.npz
        """
        config = self.datasets[label]
        trajs_xyz = []
        it_start = folder_pattern[1][0]
        it_end = folder_pattern[1][1]
        inputFolder = os.path.join(config['folder'], folder_pattern[0])
        cache_file = os.path.join(self.cache_path, f'{label}_trajs')

        try:
            cached_data = np.load(cache_file + ".npz", allow_pickle=True)
            print(f"Using cached data: {cache_file}.npz")
            self.datasets[label]['trajectories'] = cached_data['T']
            return 
        except FileNotFoundError:
            print(f"No cached data, creating cache file {cache_file}")

        for i in range(it_start, it_end + 1):
            traj = self.__load_and_process_trajectory(folder=inputFolder, replica=i, filename=filename)
            if traj.size > 0:
                trajs_xyz.append(traj)

        if trajs_xyz:
            max_shape = np.max([traj.shape for traj in trajs_xyz], axis=0)
            trajs_xyz = [self.compute_helpers.pad_array(traj, max_shape) if not np.array_equal(traj.shape, max_shape) else traj for traj in trajs_xyz]
            self.datasets[label]['trajectories'] = np.vstack(trajs_xyz)
            if cache_trajs:
                print(f"Saving trajectory cache to file {cache_file}.npz")
                np.savez_compressed(cache_file, T=np.vstack(trajs_xyz))
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
                'cmap':'viridis',
                'title': f'{args}',
            }
        self.plot_helper.plot(plot_type="dendrogram", data=Z, plot_params=plot_params)
        return Z

    def dist_map(self, label: str, metric="euclidean", norm=None, max_frames=None, linkage_method=None, size=5) -> np.array:
        """_dist_map_
        Creates the distance map for a given dataset using the specified metric, with optional normalization and linkage.
        Args:
        label (str): Label of the dataset to create the distance map for.
        metric (str): The distance metric to use. Options: "euclidean", "contact", "pearson", "spearman", "log2_contact".
        norm (str): Normalization method to apply after distance calculation.
        linkage_method (str): Linkage method to apply after normalization. If None, no linkage is performed.
        max_frames (int): Maximum number of frames to process (None for all frames).
        Returns:
        np.array: The distance map.
        """
        try:
            trajectories = self.datasets[label]['trajectories']
            if trajectories is None or len(trajectories) == 0:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return np.array([])

            print(f"Processing {len(trajectories)} trajectories...")
            
            if max_frames is not None:
                trajectories = trajectories[:max_frames]
                print(f"Limited to {max_frames} frames.")

            compute_dist = []
            for i, val in enumerate(trajectories):
                print(f"Processing frame {i+1}/{len(trajectories)}...")
                dist = self.compute_helpers.calc_dist(val, metric)
                if norm is not None:
                    dist = self.compute_helpers.norm_distMatrix(dist, norm)
                compute_dist.append(dist)

            compute_dist = np.array(compute_dist)
            print(f"Distance computation complete. Shape: {compute_dist.shape}")

            linkage_result = None
            if linkage_method is not None:
                print(f"Performing linkage with method: {linkage_method}")
                try:
                    linkage_result = self.compute_helpers.perform_linkage(compute_dist, method=linkage_method) if linkage_method else None
                    print("Linkage completed successfully.")
                except Exception as e:
                    print(f"Error in linkage: {str(e)}")
            
            if self.showPlots:
                self.plot_helper.plot(plot_type='distmap', data=compute_dist,  plot_params={'method': linkage_method, 'norm':norm, 'metric':metric, 'label':label, 'outPath':self.outPath, 'size':int(size)} )

            return compute_dist

        except Exception as e:
            print(f"An error occurred in dist_map: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([])


    def pca(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', 
            n_components: int = 2, n_clusters: int = 5, labels: list = None, plot_params: dict = None, **kwargs) -> tuple:
        """
        Performs PCA on the datasets and returns the principal components and explained variance.

        Args:
            *args: (str): The labels to create the PCA from.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.
            norm (str, optional): The normalization method to use. Default is 'ice'.
            n_components (int, optional): Number of components to keep. Default is 2.
            n_clusters (int, optional): Number of clusters for coloring. Default is 5.

        Returns:
            tuple: (np.array, np.array, np.array) The principal components, the explained variance ratio, and the components.
        """
        pcaPath = os.path.join(self.outPath, 'PCA')
        os.makedirs(pcaPath, exist_ok=True)
                
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components = self.compute_helpers.find_optimal_clusters(X, 15)
    
        
        
        pca_result, explained_variance_ratio, components = self.compute_helpers.run_reduction('pca', X, n_components)
                
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"n_components to reach 95% variance ratio {n_components_95}")
        if self.showPlots:
            extra_params = {
                'explained_variance_ratio': explained_variance_ratio,
                'components': components,
                'n_components_95': np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'reduction', 'pca', args, method, metric, norm, None, 
                                    n_clusters, n_components, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="pcaplot", data=(pca_result, explained_variance_ratio, components), plot_params=params)
        
        return pca_result, explained_variance_ratio, components

    def tsne(self, *args: str, sample_size: int = 5000, n_clusters: int = 5, n_components: int = 2, 
            method: str = 'weighted', metric: str = "euclidean", norm: str = 'ice', labels: list = None,
            plot_params: dict = None, **kwargs) -> tuple:
        """
        Performs t-SNE on the datasets and returns the t-SNE results.

        Args:
            *args: (str): The labels to create the t-SNE from.
            tsneParams (dict, optional): Parameters for the t-SNE algorithm. Default is None.
            sample_size (int, optional): The sample size for t-SNE. Default is 5000.
            n_clusters (int, optional): Number of clusters for coloring. Default is 5.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use. Default is 'euclidean'.
            norm (str, optional): The normalization method to use. Default is 'ice'.

        Returns:
            tuple: (np.array, float) The t-SNE results and the KL divergence.
        """
        tsnePath = os.path.join(self.outPath, 't-sne')
        os.makedirs(tsnePath, exist_ok=True)
        
        if n_clusters == -1:
            n_clusters = self.compute_helpers.find_optimal_clusters(X, 15)
        
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        tsne_result, kl_divergence, _ = self.compute_helpers.run_reduction('tsne', X, n_components=n_components)
         
        cumulative_variance = np.cumsum(kl_divergence)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"n_components to reach 95% variance ratio {n_components_95}")
        
        if self.showPlots:
            extra_params = {
                'kl_divergence': kl_divergence,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'reduction', 'tsne', args, method, metric, norm, sample_size, 
                                    n_clusters, n_components, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="tsneplot", data=(tsne_result, kl_divergence, None), plot_params=params)
        
        return tsne_result, kl_divergence

    def umap(self, *args: str, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', 
            norm: str = 'ice', n_components: int = 2, n_clusters: int = 5, labels: list = None, 
            plot_params: dict = None, **kwargs) -> tuple:
        """
        Performs UMAP on the datasets and returns the UMAP results.

        Args:
            *args: (str): The labels to create the UMAP from.
            umapParams (dict, optional): Parameters for the UMAP algorithm. Default is None.
            sample_size (int, optional): The sample size for UMAP. Default is 5000.
            n_clusters (int, optional): Number of clusters for coloring. Default is 5.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use. Default is 'euclidean'.
            norm (str, optional): The normalization method to use. Default is 'ice'.

        Returns:
            tuple: (np.array,) The UMAP results.
        """
        umapPath = os.path.join(self.outPath, 'UMAP')
        os.makedirs(umapPath, exist_ok=True)
        
        if n_clusters == -1:
            n_clusters = self.compute_helpers.find_optimal_clusters(X, 15)
        X, _ = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
                       
        umap_result, embedding, graph = self.compute_helpers.run_reduction('umap', X, n_components=n_components)
        print(f"UMAP result shape: {umap_result.shape}")
                 
        cumulative_variance = np.cumsum(embedding)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"n_components to reach 95% variance ratio {n_components_95}")
    
        if self.showPlots:
            extra_params = { 'embedding': embedding}
            params = self.plot_helper.fetch_params(self.outPath, 'reduction', 'umap', args, method, metric, norm, sample_size, 
                                    n_clusters, n_components, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="umapplot", data=(umap_result, embedding, graph), plot_params=params)
        
        return umap_result, embedding, graph
    
    def svd(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', 
            n_components: int = 2, n_clusters: int = 2, labels: list = None, plot_params: dict = None, **kwargs) -> tuple:
        svdPath = os.path.join(self.outPath, 'SVD')     
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
        svdPath = os.path.join(self.outPath, 'SVD')
        os.makedirs(svdPath, exist_ok=True)
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components = self.compute_helpers.find_optimal_clusters(X)
        svd_result, singular_values, vt = self.compute_helpers.run_reduction('svd', X, n_components)
            
        if self.showPlots:
                extra_params = {
                    'singular_values': singular_values,
                }
                params = self.plot_helper.fetch_params(self.outPath, 'reduction', 'svd', args, method, metric, norm, None, 
                                        n_clusters, n_components, labels, extra_params, plot_params)
                self.plot_helper.plot(plot_type="svdplot", data=(svd_result, singular_values, vt), plot_params=params)
            
        return svd_result, singular_values, vt

    def mds(self, *args: str, sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', 
            norm: str = 'ice', n_components: int = 2, n_clusters: int = 2, labels: list = None, plot_params: dict = None, **kwargs) -> tuple:
        mdsPath = os.path.join(self.outPath, 'MDS')
        os.makedirs(mdsPath, exist_ok=True) 
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
        mdsPath = os.path.join(self.outPath, 'MDS')
        os.makedirs(mdsPath, exist_ok=True)
    
        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if n_components == -1:
            n_components = self.compute_helpers.find_optimal_clusters(X)
            
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
            
        mds_result, stress, dissimilarity_matrix = self.compute_helpers.run_reduction('mds', X, n_components)
        cumulative_variance = np.cumsum(stress)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"n_components to reach 95% variance ratio {n_components_95}")
        
        if self.showPlots:
            extra_params = {
                'stress': stress,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'reduction', 'mds', args, method, metric, norm, sample_size, 
                                    n_clusters, n_components, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="mdsplot", data=(mds_result, stress, dissimilarity_matrix), plot_params=params)
        
        return mds_result, stress, dissimilarity_matrix
    
    def spectral_clustering(self, *args: str, n_clusters: int = 5, sample_size: int = 5000, method: str = 'weighted', 
                            metric: str = 'euclidean', norm: str = 'ice', n_components: int = 2, labels: list = None, 
                            plot_params: dict = None, **kwargs) -> tuple:
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
        spectralPath = os.path.join(self.outPath, 'SpectralClustering')
        os.makedirs(spectralPath, exist_ok=True)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        if n_clusters == -1:
            n_clusters = self.compute_helpers.find_optimal_clusters(X, 20)
            print(f'found optimal clusters: {n_clusters}')
                
        cluster_result, additional_info = self.compute_helpers.run_clustering('spectral', X, n_clusters=n_clusters, n_components=n_components, **spectralParams or {})
        if self.showPlots:
            extra_params = {
                'affinity_matrix_': additional_info.get('affinity_matrix_'),
            }
            params = self.plot_helper.fetch_params(self.outPath, 'clustering', 'spectral', args, method, metric, norm, sample_size, 
                                    n_clusters, n_components, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="spectralclusteringplot", data=[X, cluster_result, additional_info], plot_params=params)
        
        return cluster_result, additional_info
    
    
    def kmeans_clustering(self, *args: str, n_clusters: int = 5, sample_size: int = 5000, method: str = 'weighted', 
                        metric: str = 'euclidean', norm: str = 'ice', labels: list = None, 
                        plot_params: dict = None, **kwargs) -> tuple:
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
        kmeansPath = os.path.join(self.outPath, 'KMeans')
        os.makedirs(kmeansPath, exist_ok=True)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        if n_clusters == -1:
            n_clusters = self.compute_helpers.find_optimal_clusters(X, 20)
        
        kmeans_result, additional_info = self.compute_helpers.run_clustering('kmeans', X, n_clusters=n_clusters, **kwargs)
        if self.showPlots:
            extra_params = {
                'inertia': additional_info.get('inertia'),
            }
            params = self.plot_helper.fetch_params(self.outPath, 'clustering', 'kmeans', args, method, metric, norm, sample_size, 
                                    n_clusters, None, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="kmeansclusteringplot", data=[X, kmeans_result, additional_info], plot_params=params)
        
        return kmeans_result, additional_info

    def dbscan_clustering(self, *args: str, eps: float = 0.5, min_samples: int = 5, sample_size: int = 5000, 
                        method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', labels: list = None, 
                        plot_params: dict = None, **kwargs) -> tuple:
        dbscanPath = os.path.join(self.outPath, 'DBSCANClustering')
        os.makedirs(dbscanPath, exist_ok=True)  
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
        dbscanPath = os.path.join(self.outPath, 'DBSCANClustering')
        os.makedirs(dbscanPath, exist_ok=True)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        dbscan_result, additional_info = self.compute_helpers.run_clustering('dbscan', X, eps=eps, method=method, min_samples=min_samples, **kwargs)
        
        if self.showPlots:
            extra_params = {
                'eps': eps,
                'min_samples': min_samples,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'clustering', 'dbscan', args, method, metric, norm, sample_size, 
                                    None, None, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="dbscanplot", data=[X, dbscan_result, additional_info], plot_params=params)
        
        return dbscan_result, additional_info

    def hierarchical_clustering(self, *args: str, n_clusters: int = 5, sample_size: int = 5000, method: str = 'ward', 
                                metric: str = 'euclidean', norm: str = 'ice', labels: list = None, plot_params: dict = None, **kwargs) -> tuple:
        hierarchicalPath = os.path.join(self.outPath, 'HierarchicalClustering')
        os.makedirs(hierarchicalPath, exist_ok=True) 
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
        hierarchicalPath = os.path.join(self.outPath, 'HierarchicalClustering')
        os.makedirs(hierarchicalPath, exist_ok=True)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        hierarchical_result, additional_info = self.compute_helpers.run_clustering('hierarchical', X, n_clusters=n_clusters, method=method, **kwargs)
        
        if self.showPlots:
            extra_params = {
                'linkage_method': method,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'clustering', 'hierarchical', args, method, metric, norm, sample_size, 
                                    n_clusters, None, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="hierarchicalplot", data=[X, hierarchical_result, additional_info], plot_params=params)
        
        return hierarchical_result, additional_info

    def optics_clustering(self, *args: str, min_samples: int = 5, xi: float = 0.05, min_cluster_size: float = 0.05, 
                        sample_size: int = 5000, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice', 
                        labels: list = None, plot_params: dict = None, **kwargs) -> tuple:
        opticsPath = os.path.join(self.outPath, 'OPTICSClustering')
        os.makedirs(opticsPath, exist_ok=True) 
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
        opticsPath = os.path.join(self.outPath, 'OPTICSClustering')
        os.makedirs(opticsPath, exist_ok=True)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
        
        optics_result, additional_info = self.compute_helpers.run_clustering('optics', X, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, **kwargs)
        
        if self.showPlots:
            extra_params = {
                'min_samples': min_samples,
                'xi': xi,
                'min_cluster_size': min_cluster_size,
            }
            params = self.plot_helper.fetch_params(self.outPath, 'clustering', 'optics', args, method, metric, norm, sample_size, 
                                    None, None, labels, extra_params, plot_params)
            self.plot_helper.plot(plot_type="opticsplot", data=[X, optics_result, additional_info], plot_params=params)
        
        return optics_result, additional_info


    """=========================================UTILITIES========================================"""



    def calc_XZ(self, *args: str, method: str = 'none', metric: str = 'euclidean', norm: str = 'none', overrideCache: bool = False) -> tuple:
        """
        Calculate and cache the distance matrix and linkage matrix for given datasets.

        Args:
            *args (str): Labels of the datasets to process.
            method (str): The linkage method to use.
            metric (str): The distance metric to use.
            norm (str): The normalization method to use.
            overrideCache (bool): option to not fetch from cache and recompute X,Z

        Returns:
            tuple: (X, Z) where X is the flattened distance array and Z is the linkage matrix.
        """
        key = tuple(sorted(args)) + (method, metric, norm)
        cache_file = os.path.join(self.cache_path, f"cache_{key}.pkl")
        
        if self.execution_mode.getExecutionMode() == 'cuda':
            return self.compute_helpers.calc_XZ(
                datasets=self.datasets,
                args=args,
                cache_path=self.cache_path,
                method=method,
                metric=metric,
                norm=norm,
                overrideCache=overrideCache
        )
        
        # Try to load cached data
        if overrideCache == False:
            try:
                cached_data = np.load(cache_file + ".npz", allow_pickle=True)
                print(f"Using cached data: {cache_file}.npz")
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
            
            dist = self.compute_helpers.cached_calc_dist(trajectories, metric=metric)
            dist = np.array(dist)
            print(f"{label} has dist shape {dist.shape}")
            
            # Handle infinite values
            inf_mask = np.isinf(dist)
            if np.any(inf_mask):
                print(f"Warning: Infinite values found in distance matrix for {label}. Replacing with large finite value.")
                large_finite = np.finfo(dist.dtype).max / 2
                dist[inf_mask] = large_finite
            
            # Handle NaN values (likely centromere regions)
            nan_mask = np.isnan(dist)
            if np.any(nan_mask):
                print(f"Warning: NaN values found in distance matrix for {label}. These are likely centromere regions.")
                # Replace NaNs with the mean of non-NaN values
                dist[nan_mask] = np.nanmean(dist)
            
            if norm.lower() != 'none':
                normalized_dist = np.array([self.compute_helpers.norm_distMatrix(matrix=matrix, norm=norm) for matrix in dist])
            else:
                normalized_dist = dist
                
            flat_euclid_dist_map[label] = normalized_dist

            
            max_shape = np.maximum(max_shape, np.max([d.shape for d in normalized_dist], axis=0))
        
        # Pad arrays to ensure consistent shapes
        padded_flat_euclid_dist_map = {
            label: [np.pad(val, ((0, max_shape[0] - val.shape[0]), (0, max_shape[1] - val.shape[1]))) 
                    for val in sublist] 
            for label, sublist in flat_euclid_dist_map.items()
        }
        
        # Flatten and stack distance matrices
        flat_euclid_dist_map = {
            label: [padded_flat_euclid_dist_map[label][val][np.triu_indices_from(padded_flat_euclid_dist_map[label][val], k=1)].flatten()
                    for val in range(len(padded_flat_euclid_dist_map[label]))]
            for label in args
        }
        
        X = np.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        # check for non-finite values
        non_finite_mask = ~np.isfinite(X)
        if np.any(non_finite_mask):
            print("Warning: Non-finite values found in flattened distance array. Replacing with mean value.")
            X[non_finite_mask] = np.nanmean(X)
        
        if method.lower() != 'none' or method.lower() != 'weighted':
            print(f"Preprocessing X with {method}")
            X = self.compute_helpers.preprocess_X(X, method)
            # recheck after preprocessing x
            X = self.compute_helpers.preprocess_check(X)
            if np.any(non_finite_mask):
                print("Warning: Non-finite values found in flattened distance array. Replacing with mean value.")
                X[non_finite_mask] = np.nanmean(X)
        
        # Perform linkage
        if method.lower() == 'remove':
            try:
                Z = linkage(X, method='weighted', metric='euclidean')
            except ValueError as e:
                print(f"Error in linkage: {e}")
                print("Attempting to proceed with available finite values...")
                # Create a mask for finite values
                finite_mask = np.isfinite(X)
                X_finite = X[finite_mask]
                Z = linkage(X_finite, method='weighted', metric='euclidean')
            
        # Cache the results
        Z = np.array(1)
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
    
    def setExecutionMode(self, execution_mode, **kwargs):
        from AnalysisTools.CompHelperCPU import ComputeHelpersCPU
        
        if isinstance(execution_mode, dict):
            mode = execution_mode.get("mode", "cpu")
            params = execution_mode.get("execParams", {})
        elif isinstance(execution_mode, ComputeHelpersCPU):
            mode = "custom"
            self.compute_helpers = execution_mode
            self.execution_mode == 'cpu'
        else:
            mode = execution_mode
            params = kwargs
            
        if mode.lower() == "gpu" or mode.lower() == "cuda":
            from AnalysisTools.CompHelperGPU import ComputeHelpersGPU
            self.compute_helpers = ComputeHelpersGPU(**params)
            self.execution_mode == 'cuda'
        elif mode.lower() == "cpu":
            self.compute_helpers = ComputeHelpersCPU(**params)
            self.execution_mode == 'cpu'
        elif mode!= "custom":
            raise ValueError("Invalid execution mode. Use 'cpu', 'gpu', or pass a ComputeHelpers instance.")
        
    
    def setShowPlots(self, show: bool):
        self.showPlots = show