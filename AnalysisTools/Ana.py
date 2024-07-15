import AnalysisTools.compute_helpers as compute_helpers
from AnalysisTools.plot_helpers import PlotHelper
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
    def __init__(self, outputFolderPath: str = "", execution_mode: str = 'cpu', showPlots=True):
        """
        Initializes the Ana class with a base folder for data storage.
        """
        self.datasets = {}
        self.cndbTools = cndbTools()
        self.execution_mode = execution_mode
        self.showPlots = showPlots
        
        if self.showPlots:
            self.plot_helper = PlotHelper()
        else:
            self.plot_helper = None
        
        if outputFolderPath is "":
            self.outPath = os.path.join(os.getcwd(), 'Analysis')
            os.makedirs(self.outPath, exist_ok=True)
        else:
            self.outPath = os.path.join(os.getcwd(), outputFolderPath)
            os.makedirs(self.outPath, exist_ok=True)
            
        self.cache_path = os.path.join(os.getcwd(), 'cache')
        os.makedirs(self.cache_path, exist_ok=True)
        
        self.execution_mode = execution_mode
        if self.execution_mode == 'cuda':
            try:
                import cupy
                compute_helpers.set_cuda_availability(True)
                print("CUDA is available and will be used for computations.")
            except ImportError:
                print("CUDA requested but CuPy not found. Falling back to CPU.")
                self.execution_mode = 'cpu'
                compute_helpers.set_cuda_availability(False)
        else:
            compute_helpers.set_cuda_availability(False)


    def add_dataset(self, label: str, folder: str):
        """
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
            trajs_xyz = [compute_helpers.pad_array(traj, max_shape) if not np.array_equal(traj.shape, max_shape) else traj for traj in trajs_xyz]
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

        if trajectories is None:
            print(f"Trajectories not yet loaded for {label}. Load them first")
            return np.array([])

        if self.datasets[label]["distance_array"] is None:
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

    def pca(self, *args: str, method: str = 'weighted', metric: str = 'euclidean', norm: str = 'ice') -> tuple:
        """
        Performs PCA on the datasets and returns the principal components and explained variance.

        Args:
            *args: (str): The labels to create the PCA from.
            method (str, optional): The method for hierarchical clustering. Default is 'weighted'.
            metric (str, optional): The distance metric to use for computing pairwise distances. Default is 'euclidean'.

        Returns:
            tuple: (np.array, np.array) The principal components and the explained variance ratio.
        """
        num_clusters = len(args)
        flattened_distance_array, linkage_matrix = self.calc_XZ(*args, method=method, metric=metric, norm=norm)

        threshold = linkage_matrix[-num_clusters, 2]
        fclust = fcluster(linkage_matrix, t=threshold, criterion='distance')

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(flattened_distance_array)
        explained_variance_ratio = pca.explained_variance_ratio_
        
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'pca_plot_{args}-{method}_{metric}.png'),
                'title': 'PCA Plot',
                'x_label': 'Principal Component 1',
                'y_label': 'Principal Component 2'
            }
        self.plot_helper.plot(plot_type="pcaplot", 
                              data=[principalComponents, explained_variance_ratio, fclust], 
                              plot_params=plot_params)

        return principalComponents, explained_variance_ratio, fclust

    def tsne(self, *args: str, tsneParams: dict = None, sample_size: int = 5000, num_clusters: int = -1, method: str = 'weighted', metric:str = "euclidean", norm: str = 'ice') -> tuple:
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
        if num_clusters is -1:
            num_clusters = len(args)

        default_tsne_params = {
            'n_components': 2,
            'verbose': 1,
            'max_iter': 800,
        }

        if tsneParams is not None:
            default_tsne_params.update(tsneParams)

        X, Z = self.calc_XZ(*args, method=method,metric=metric, norm=norm)

        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)

        n_samples = X.shape[0]
        perplexity = min(30, max(5, n_samples // 10))
        default_tsne_params['perplexity'] = perplexity

        threshold = Z[-num_clusters, 2]
        fclust = fcluster(Z, t=threshold, criterion='distance')

        tsne = TSNE(**default_tsne_params)
        tsne_res = tsne.fit_transform(X)
        
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'tsne_plot_{args}_{method}.png'),
                'title': 't-SNE Plot',
                'x_label': 't-SNE 1',
                'y_label': 't-SNE 2'
            }
        self.plot_helper.plot(plot_type="tsneplot", data=[tsne_res, fclust], plot_params=plot_params)

        return tsne_res, fclust

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
        if num_clusters is -1:
            num_clusters = len(args)

        default_umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'random_state': 42
        }

        if umapParams is not None:
            default_umap_params.update(umapParams)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)

        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)

        threshold = Z[-num_clusters, 2]
        fclust = fcluster(Z, t=threshold, criterion='distance')

        reducer = umap.UMAP(**default_umap_params)
        umap_res = reducer.fit_transform(X)
        
        
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'umap_plot_{args}_{method}.png'),
                'title': 'UMAP Plot',
                'x_label': 'UMAP 1',
                'y_label': 'UMAP 2'
            }
        self.plot_helper.plot(plot_type="umapplot", data=[umap_res, fclust], plot_params=plot_params)


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
        if num_clusters is -1:
            num_clusters = len(args)

        default_ivis_params = {
            'embedding_dims': 2,
            'k': 15,
            'epochs': 10,
            'batch_size': 128,
            'verbose': 1,
        }

        if ivisParams is not None:
            default_ivis_params.update(ivisParams)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)

        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)

        threshold = Z[-num_clusters, 2]
        fclust = fcluster(Z, t=threshold, criterion='distance')

        if X.shape[0] < default_ivis_params['batch_size']:
            default_ivis_params['batch_size'] = X.shape[0]

        model = ivis.Ivis(**default_ivis_params)
        ivis_res = model.fit_transform(X)
        
        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'ivis_plot_{args}_{method}.png'),
                'title': 'IVIS Plot',
                'x_label': 'IVIS 1',
                'y_label': 'IVIS 2'
            }
        self.plot_helper.plot(plot_type="ivisplot", data=[ivis_res, fclust], plot_params=plot_params)


        return ivis_res, fclust
    
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
        default_spectral_params = {
            'n_neighbors': 30,
            'random_state': 42,
            'assign_labels': 'kmeans'
        }

        if spectralParams is not None:
            default_spectral_params.update(spectralParams)

        X, Z = self.calc_XZ(*args, method=method, metric=metric, norm=norm)

        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)

        if num_clusters is -1:
            num_clusters = len(args)

        # Improve affinity matrix calculation
        n_neighbors = min(default_spectral_params['n_neighbors'], X.shape[0] - 1)
        knn_graph = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X).kneighbors_graph()
        affinity_matrix = 0.5 * (knn_graph + knn_graph.T)
        
        # Use Gaussian kernel for affinity
        sigma = np.mean(affinity_matrix.data)
        affinity_matrix.data = np.exp(-affinity_matrix.data ** 2 / (2. * sigma ** 2))
        
        # Perform spectral clustering
        sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', 
                                random_state=default_spectral_params['random_state'], 
                                assign_labels=default_spectral_params['assign_labels'])
        cluster_labels = sc.fit_predict(affinity_matrix)
        
        # Compute Laplacian and its eigenvectors
        laplacian = csgraph.laplacian(affinity_matrix, normed=True)
        eigenvalues, eigenvectors = eigsh(laplacian, k=num_clusters, which='SM')
        
        # Compute clustering metrics
        silhouette_avg, calinski_harabasz, davies_bouldin = compute_helpers.evaluate_clustering(X, cluster_labels)

        results = (cluster_labels, eigenvalues, eigenvectors, affinity_matrix, 
                silhouette_avg, calinski_harabasz, davies_bouldin)

        if self.showPlots:
            plot_params = {
                'outputFileName': os.path.join(self.outPath, f'spectral_clustering_plot_{args}_{method}_{metric}.png'),
                'title': 'Spectral Clustering Plot',
            }
            self.plot_helper._spectralclusteringplot(X, results, plot_params)

        return results

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
            
            dist = compute_helpers.cached_calc_dist(trajectories, metric, self.execution_mode, n_jobs)
            dist = np.array(dist)
            print(f"{label} has dist shape {dist.shape}")
            
            normalized_dist = np.array([compute_helpers.norm_distMatrix(matrix, norm=norm) for matrix in dist])
            self.datasets[label]["distance_array"] = normalized_dist
            flat_euclid_dist_map[label] = normalized_dist
            
            max_shape = np.maximum(max_shape, np.max([d.shape for d in normalized_dist], axis=0))
        
        padded_flat_euclid_dist_map = {label: [np.pad(val, ((0, max_shape[0] - val.shape[0]), (0, max_shape[1] - val.shape[1]))) for val in sublist] for label, sublist in flat_euclid_dist_map.items()}
        
        flat_euclid_dist_map = {label: [padded_flat_euclid_dist_map[label][val][np.triu_indices_from(padded_flat_euclid_dist_map[label][val], k=1)].flatten()
                                        for val in range(len(padded_flat_euclid_dist_map[label]))]
                                for label in args}
        
        X = np.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        Z = linkage(X, method=method, metric='euclidean')
        
        np.savez_compressed(cache_file, X=X, Z=Z)
        return X, Z