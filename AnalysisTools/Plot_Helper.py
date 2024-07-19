import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from AnalysisTools.compute_helpers import ComputeHelpers
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
        self.compute_helpers = ComputeHelpers()
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
        
        fig = plt.figure(figsize=params['figsize'])
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(result[:, 0], result[:, 1], alpha=params['alpha'], s=params['size'], cmap=params['cmap'])
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(result[:, 0], result[:, 1], result[:, 2], alpha=params['alpha'], cmap=params['cmap'], s=params['size'])
            ax.set_zlabel(params['z_label'])
        
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        plt.title(params['title'])
        
        plt.colorbar(scatter)
        
        if additional_info is not None:
            if isinstance(additional_info, np.ndarray) and additional_info.size == n_components:
                for i, var in enumerate(additional_info):
                    plt.text(0.05, 0.95 - i*0.05, f"PC{i+1} variance: {var:.2f}", transform=ax.transAxes)
            elif isinstance(additional_info, float):
                plt.text(0.05, 0.95, f"Additional info: {additional_info:.4f}", transform=ax.transAxes)
        
        print(f"Dimensionality reduction plot created with {n_components} components")
        self._save_and_show(params)

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

    def _ivisplot(self, data, params):
        """
        Create an ivis plot.

        Args:
            data (list): [ivis_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        print("\nivis Plot:")
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
        result, singular_values, vt = data
        n_components = params.get('n_components', 2)
        
        fig = plt.figure(figsize=params['figsize'])
        
        # Plot the reduced data
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(result[:, 0], result[:, 1], alpha=params['alpha'], s=params['size'], cmap=params['cmap'])
        ax1.set_xlabel(params['x_label'])
        ax1.set_ylabel(params['y_label'])
        ax1.set_title('SVD Reduced Data')
        plt.colorbar(scatter, ax=ax1)
        
        # Plot the singular values
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, len(singular_values) + 1), singular_values, 'bo-')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Singular Value')
        ax2.set_title('Singular Values')
        
        plt.tight_layout()
        plt.suptitle(params['title'], fontsize=16)
        
        print(f"SVD plot created with {n_components} components")
        self._save_and_show(params)

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
        result, stress, _ = data
        n_components = params.get('n_components', 2)
        
        fig = plt.figure(figsize=params['figsize'])
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(result[:, 0], result[:, 1], alpha=params['alpha'], s=params['size'], cmap=params['cmap'])
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(result[:, 0], result[:, 1], result[:, 2], alpha=params['alpha'], cmap=params['cmap'], s=params['size'])
            ax.set_zlabel(params['z_label'])
        
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        plt.title(f"{params['title']} (Stress: {stress:.4f})")
        
        plt.colorbar(scatter)
        
        print(f"MDS plot created with {n_components} components")
        self._save_and_show(params)
        
 
    def _spectralclusteringplot(self, data, cluster_results, params):
        """
        Create a comprehensive spectral clustering plot.

        Args:
            data (np.array): The original data.
            cluster_results (tuple): Results from spectral_clustering method.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        (cluster_labels, eigenvalues, eigenvectors, affinity_matrix, 
         silhouette_avg, calinski_harabasz, davies_bouldin) = cluster_results
        
        n_clusters = len(np.unique(cluster_labels))

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)

        # Plot spectral embedding
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c=cluster_labels, cmap='viridis')
        ax1.set_title('Spectral Embedding')
        ax1.set_xlabel('First eigenvector')
        ax1.set_ylabel('Second eigenvector')

        # Plot original data with clustering results
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis')
        ax2.set_title('Clustered Data')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')

        # Plot eigenvalues
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        ax3.set_title('Eigenvalues')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Eigenvalue')

        # Plot silhouette
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_silhouette(ax4, data, cluster_labels, n_clusters, silhouette_avg)

        # Plot affinity matrix
        ax5 = fig.add_subplot(gs[1, 1])
        sns.heatmap(affinity_matrix.toarray(), ax=ax5, cmap='viridis')
        ax5.set_title('Affinity Matrix')

        # Plot first few eigenvectors
        ax6 = fig.add_subplot(gs[1, 2])
        for i in range(min(5, n_clusters)):
            ax6.plot(eigenvectors[:, i], label=f'Eigenvector {i+1}')
        ax6.legend()
        ax6.set_title('First Few Eigenvectors')
        ax6.set_xlabel('Data point index')
        ax6.set_ylabel('Eigenvector value')

        # Display metrics
        metrics_text = (f"Silhouette Score: {silhouette_avg:.4f}\n"
                        f"Calinski-Harabasz Score: {calinski_harabasz:.4f}\n"
                        f"Davies-Bouldin Score: {davies_bouldin:.4f}")

        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        ax7.text(0.1, 0.7, metrics_text, fontsize=12, verticalalignment='top')

        plt.tight_layout()
        self._save_and_show(params)
        
    def _plot_silhouette(self, ax, data, cluster_labels, n_clusters, silhouette_avg):
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
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