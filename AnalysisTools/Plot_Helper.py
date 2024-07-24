import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from AnalysisTools.Comp_Helper_CPU import ComputeHelpersCPU
import os

class PlotHelper:
    """PlotHelper
    Generate a specified type of plot.

    Args:
        plot_type (str): Type of plot to generate.
        data (list): Data to be plotted.
        plot_params (dict, optional): Custom plotting parameters.
        **kwargs: Additional keyword arguments for the specific plot type.

    Returns:
        None
    """
    def __init__(self, cacheStorage: str = '', AnalysisStorage: str = '' ):
        self.default_params = {
            'figsize': (10, 7),
            'cmap': 'viridis',
            'x_label': 'X-axis',
            'y_label': 'Y-axis',
            'z_label': 'Z-axis',
            'title': 'Plot',
            'size': 50,
            'alpha': 0.5,
            'linestyle': '-',
            'marker': 'o',
            'color': 'b',
            'outputFileName': 'plot.png',
            'show': True,
            'vmin': 0.001,
            'vmax': 1.0,
            'colors': ['tab:orange', 'tab:green', 'tab:red'],
            'linestyles': ('-', '--'),
            'linewidths': (2, 2),
            'labels': ('Label 1', 'Label 2'),
            'fraction': 0.046,
            'n_components': 2,
            'max_clusters': 10
        }
        
        self.cacheStorage = os.path.join(os.getcwd(), cacheStorage)
        self.compute_helpers = ComputeHelpersCPU()
        self.compute_helpers.setMem(path=self.cacheStorage)
        
        if AnalysisStorage == '':
            self.AnalysisStorage = os.path.join(os.getcwd(), 'Analysis')
            os.makedirs(self.AnalysisStorage, exist_ok=True)
        else:
            self.AnalysisStorage = os.path.join(os.getcwd(), AnalysisStorage)
            os.makedirs(self.AnalysisStorage, exist_ok=True)

    def plot(self, plot_type, data, plot_params=None, **kwargs):
        if plot_params:
            params = {**self.default_params, **plot_params}
        else:
            params = self.default_params.copy()
        
        plot_func = getattr(self, f"_{plot_type.lower()}", None)
        if plot_func:
            print(f"\nGenerating {plot_type} plot...")
            plot_func(data, params, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _hicmapplot(self, data, params):
        """
        Create a HiC map plot.

        Args:
            data (list): [hic_exp, hic_sim] matrices.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        hic_exp, hic_sim = data
        comp = np.triu(hic_exp) + np.tril(hic_sim, k=1)
        plt.figure(figsize=params['figsize'])
        plt.matshow(comp, norm=plt.colors.LogNorm(vmin=params['vmin'], vmax=params['vmax']), cmap=params['cmap'])
        plt.title(params['title'])
        plt.colorbar()
        print(f"HiC map plot created with vmin={params['vmin']}, vmax={params['vmax']}")
        self._save_and_show(params)


    def _genomedistanceplot(self, data, params):
        """
        Create a genome distance plot.

        Args:
            data (list): [scale_exp, scale_sim] arrays.
            params (dict): Plotting parameters.

        Returns:
            None
        """

        scale_exp, scale_sim = data
        fig, ax = plt.subplots(figsize=params['figsize'])
        ax.loglog(range(len(scale_exp)), scale_exp, color=params['colors'][0], label=params['labels'][0], linestyle=params['linestyles'][0], linewidth=params['linewidths'][0])
        ax.loglog(range(len(scale_sim)), scale_sim, color=params['colors'][1], linestyle=params['linestyles'][1], label=params['labels'][1], linewidth=params['linewidths'][1])
        ax.set_title(params['title'])
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        ax.legend()
        print(f"Genome distance plot created with {len(scale_exp)} data points")
        self._save_and_show(params)


    def _errorplot(self, data, params):
        """
        Create an error plot.

        Args:
            data (np.ndarray): Array of error values.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        iterations = np.arange(len(data))
        plt.figure(figsize=params['figsize'])
        plt.plot(iterations, data, marker=params['marker'], linestyle=params['linestyle'], color=params['color'])
        plt.xlabel(params['x_label'])
        plt.ylabel(params['y_label'])
        plt.title(params['title'])
        plt.grid(True)
        plt.legend()
        print(f"Error plot created with {len(data)} data points")
        print(f"Min error: {min(data):.4f}, Max error: {max(data):.4f}")
        self._save_and_show(params)

    def _dendrogram(self, data, params):
        """
        Create a dendrogram.

        Args:
            data (np.ndarray): Hierarchical clustering or linkage matrix.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        plt.figure(figsize=params['figsize'])
        hierarchy.dendrogram(data)
        plt.title(params['title'])
        print("Dendrogram created")
        self._save_and_show(params)

    def _euclidiandistmap(self, data, params):
        """
        Create a Euclidean distance map.

        Args:
            data (list): [distance_matrix] containing Euclidean distances.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        fig, ax = plt.subplots(1, 1, figsize=params['figsize'])
        p = ax.imshow(data[0], vmin=params['vmin'], vmax=params['vmax'], cmap=params['cmap'])
        plt.colorbar(p, ax=ax, fraction=params['fraction'])
        plt.title(params['title'])
        plt.xlabel(params['x_label'])
        plt.ylabel(params['y_label'])
        print(f"Euclidean distance map created with shape {data[0].shape}")
        print(f"Min distance: {np.min(data[0]):.4f}, Max distance: {np.max(data[0]):.4f}")
        self._save_and_show(params)


    def _dimensionality_reduction_plot(self, data, params):
        result, additional_info, _ = data
        n_components = params.get('n_components', 2)
        plot_type = params.get("plot_type")

        # Perform clustering and evaluation
        cluster_labels, _ = self.compute_helpers.evaluate_clustering(result, n_clusters=params.get('n_clusters', 5))
        
        fig = plt.figure(figsize=params['figsize'])
        
        if n_components == 1:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(range(len(result)), result[:, 0],
                                c=range(len(result)),
                                alpha=params['alpha'],
                                s=params['size'],
                                cmap=params['cmap'])
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(params['y_label'])
        elif n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(result[:, 0], result[:, 1],
                                c=range(len(result)),
                                alpha=params['alpha'],
                                s=params['size'],
                                cmap=params['cmap'])
            ax.set_xlabel(params['x_label'])
            ax.set_ylabel(params['y_label'])
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(result[:, 0], result[:, 1], result[:, 2],
                                c=range(len(result)),
                                alpha=params['alpha'],
                                s=params['size'],
                                cmap=params['cmap'])
            ax.set_xlabel(params['x_label'])
            ax.set_ylabel(params['y_label'])
            ax.set_zlabel(params['z_label'])

        plt.title(params['title'])
        plt.colorbar(scatter, label='Sample Index')
        
        # Prepare logging information
        log_info = f"""
            {'='*50}
            File: {params.get('outputFileName')}
            {'='*50}
            Parameters:
            - Plot Type: {plot_type}
            - n_components: {n_components}
            - n_clusters: {params.get('n_clusters')}
            - method: {params.get('method')}
            - metric: {params.get('metric')}
            - norm: {params.get('norm')}

            Results:
            """

        
    # Handle additional info based on the reduction method
        if plot_type == 'pcaplot':
            if isinstance(additional_info, np.ndarray) and additional_info.size == n_components:
                variance_text = "Explained variance: " + ", ".join([f"PC{i+1} {var:.2%}" for i, var in enumerate(additional_info)])
                plt.text(0.05, 0.95, variance_text, transform=ax.transAxes, verticalalignment='top')
        elif plot_type == 'tsneplot':
            if isinstance(additional_info, float):
                plt.text(0.05, 0.95, f"KL divergence: {additional_info:.4f}", transform=ax.transAxes, verticalalignment='top')
        elif plot_type == 'umapplot':
            if isinstance(params.get('embedding'), np.ndarray):
                embedding = params.get('embedding')
                singular_values = embedding.flatten()[:5]  # Flatten to get a 1D array if embedding is 2D
                singular_values_text = "UMAP Embedding: " + ", ".join([f"{val:.2f}" for val in singular_values])
                plt.text(0.05, 0.95, singular_values_text, transform=ax.transAxes, verticalalignment='top')
        elif plot_type == 'svdplot':
            if isinstance(additional_info, np.ndarray):
                singular_values_text = "Top singular values: " + ", ".join([f"{val:.2f}" for val in additional_info[:5]])
                plt.text(0.05, 0.95, singular_values_text, transform=ax.transAxes, verticalalignment='top')
        elif plot_type == 'mdsplot':
            if isinstance(additional_info, float):
                plt.text(0.05, 0.95, f"Stress: {additional_info:.4f}", transform=ax.transAxes, verticalalignment='top')
        else:
            if isinstance(additional_info, float):
                plt.text(0.05, 0.95, f"Additional info: {additional_info:.4f}", transform=ax.transAxes, verticalalignment='top')
        
        # Add n_components_95 information if available
        n_components_95 = params.get('n_components_95')
        if n_components_95 is not None:
            plt.text(0.05, 0.90, f"Components for 95% variance: {n_components_95}", transform=ax.transAxes, verticalalignment='top')
        
        print(f"Dimensionality reduction plot created with {n_components} components")
        self._save_and_show(params)
        
    # Add clustering metrics
        cluster_labels, _ = self.compute_helpers.evaluate_clustering(result, n_clusters=params.get('n_clusters'))
        log_info += f"""
        Clustering Metrics:
        - Silhouette Score: {_[0]:.4f} (higher is better, range: [-1, 1])
        - Calinski-Harabasz Index: {_[1]:.4f} (higher is better)
        - Davies-Bouldin Index: {_[2]:.4f} (lower is better)

        {'='*50}
        """
                # Write to log file
        log_dir = os.path.dirname(params.get('outputFileName'))
        log_path = os.path.join(log_dir, 'dimensionality_reduction_log.txt')
        with open(log_path, "a") as f:
            f.write(log_info)

        print(log_info) 
        
    def _clustering_plot(self, data, params):
        X, labels, additional_info = data
        n_clusters = len(np.unique(labels))
        plot_type = params['plot_type']
        
        labels, _ = self.compute_helpers.evaluate_clustering(X, labels)

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)

        # Plot clustered data
        ax1 = fig.add_subplot(gs[0, :2])
        if X.shape[1] >= 2:
            scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap=params['cmap'], alpha=0.7)
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
        else:
            scatter = ax1.scatter(range(len(X)), X.ravel(), c=labels, cmap=params['cmap'], alpha=0.7)
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Feature Value')
        ax1.set_title(f'{plot_type.capitalize()} Clustering')
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # Plot silhouette
        ax2 = fig.add_subplot(gs[1, :2])
        silhouette_avg = _[0]
        self._plot_silhouette(ax2, X, labels, n_clusters, silhouette_avg)

        # Plot cluster sizes
        ax3 = fig.add_subplot(gs[2, 0])
        unique, counts = np.unique(labels, return_counts=True)
        ax3.bar(unique, counts)
        ax3.set_title('Cluster Sizes')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Samples')

        # Display metrics
        metrics_text = (f"Silhouette Score: {silhouette_avg:.4f}\n"
                        f"Calinski-Harabasz Score: {_[1]:.4f}\n"
                        f"Davies-Bouldin Score: {_[2]:.4f}")

        ax4 = fig.add_subplot(gs[:, 2])
        ax4.axis('off')
        ax4.text(0.1, 0.7, metrics_text, fontsize=12, verticalalignment='top')

        # Method-specific plots
        if plot_type == 'kmeans':
            ax5 = fig.add_subplot(gs[2, 1])
            inertias = params.get('inertias', [])  # Get list of inertias for different k
            if inertias:
                ax5.plot(range(1, len(inertias) + 1), inertias)
                ax5.set_title('Elbow Method')
                ax5.set_xlabel('Number of Clusters (k)')
                ax5.set_ylabel('Inertia')
            else:
                ax5.text(0.5, 0.5, 'Inertia data not available', ha='center', va='center')

        elif plot_type == 'spectral':
            ax5 = fig.add_subplot(gs[2, 1])

            affinity_matrix = params.get('affinity_matrix_')
            ax5.imshow(affinity_matrix, cmap=params.get('cmap'))
            ax5.set_title('Affinity Matrix')
            ax5.set_xlabel('Sample Index')
            ax5.set_ylabel('Sample Index')
 
        elif plot_type == 'hierarchical':
            from scipy.cluster.hierarchy import dendrogram
            ax5 = fig.add_subplot(gs[2, 1])
            dendrogram(additional_info['linkage_matrix'], ax=ax5, truncate_mode='lastp', p=10)
            ax5.set_title('Dendrogram')

        elif plot_type == 'dbscan':
            ax5 = fig.add_subplot(gs[2, 1])
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[additional_info['core_sample_indices']] = True
            unique, counts = np.unique(labels[core_samples_mask], return_counts=True)
            ax5.bar(unique, counts)
            ax5.set_title('Core Point Distribution')
            ax5.set_xlabel('Cluster')
            ax5.set_ylabel('Number of Core Points')

        elif plot_type == 'optics':
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(additional_info['reachability'])
            ax5.set_title('Reachability Plot')
            ax5.set_xlabel('Points')
            ax5.set_ylabel('Reachability Distance')

        plt.tight_layout()
        self._save_and_show(params)

        # Prepare logging information
        log_info = f"""
        {'='*50}
        File: {params.get('outputFileName')}
        {'='*50}
        Parameters:
        - Plot Type: {plot_type}
        - Number of Clusters: {n_clusters}
        - Method: {params.get('method')}
        - Metric: {params.get('metric')}
        - Norm: {params.get('norm')}

        Results:
        {metrics_text}
        """

        # Write to log file
        log_dir = os.path.dirname(params.get('outputFileName'))
        log_path = os.path.join(log_dir, 'clustering_log.txt')
        with open(log_path, "a") as f:
            f.write(log_info)

        print(log_info)

    def _plot_silhouette(self, ax, X, labels, n_clusters, silhouette_avg):
        sample_silhouette_values = silhouette_samples(X, labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
        ax.set_title("Silhouette plot for the various clusters")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
            
    def _pcaplot(self, data, params):
        """
        Create a PCA plot.

        Args:
            data (list): [principalDF, explained_variance_ratio, fclust]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nPCA Plot:")
        self._dimensionality_reduction_plot(data, params)


    def _tsneplot(self, data, params):
        """
        Create a t-SNE plot.

        Args:
            data (list): [tsne_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nt-SNE Plot:")
        self._dimensionality_reduction_plot(data, params)
        
    def _umapplot(self, data, params):
        """
        Create a UMAP plot.

        Args:
            data (list): [umap_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nUMAP Plot:")
        self._dimensionality_reduction_plot(data, params)
        
 
    def _svdplot(self, data, params):
        """
        Create an SVD plot.

        Args:
            data (tuple): (svd_result, singular_values, vt) The SVD results, singular values, and right singular vectors.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nSVD Plot:")
        self._dimensionality_reduction_plot(data, params)

    def _mdsplot(self, data, params):
        """
        Create an MDS plot.

        Args:
            data (tuple): (mds_result, stress, dissimilarity_matrix) The MDS results, stress value, and dissimilarity matrix.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nMDS Plot:")
        self._dimensionality_reduction_plot(data, params)
 
    def _spectralclusteringplot(self, data, params):
        """
        Create a comprehensive spectral clustering plot.

        Args:
            data (np.array): The original data.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        return self._clustering_plot(data, params)
    
    def _kmeansclusteringplot(self, data, params):
        """_kmeansclusteringplot 

        Create a comprehensive spectral clustering plot.

        Args:
            data (np.array): The original data.
            params (dict): Plotting parameters.
        """
        return self._clustering_plot(data, params)
        
 
        
    def _save_and_show(self, params):
        """
        Save the current plot to a file and optionally display it.

        Args:
            params (dict): Plotting parameters including 'outputFileName' and 'show'.

        Returns:
            None
        """
        plt.savefig(params['outputFileName'])
        print(f"\nPlot saved as: {params['outputFileName']}")
        # if params['show']:
        plt.show()
        plt.close()
        
        
    """======================================================== Getters / Setters ========================================================"""
    def getInitialParams(self):
        return self.default_params
    
    def setInitialParams(self, params: dict):
        for key in params:
            self.default_params[key] = params[key]
        print(f'the updated initial plot parameters are: {self.default_params}')
        
    def setMeMForComputeHelpers(self, path: str):
        self.compute_helpers.setMem(path=path)