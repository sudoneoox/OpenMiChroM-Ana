import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import seaborn as sns
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator

class PlotHelper:
    """
    Generate a specified type of plot.

    Args:
        plot_type (str): Type of plot to generate.
        data (list): Data to be plotted.
        plot_params (dict, optional): Custom plotting parameters.
        **kwargs: Additional keyword arguments for the specific plot type.

    Returns:
        None
    """
    def __init__(self):
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

    def _find_optimal_clusters(self, data):
        """
        Find the optimal number of clusters.

        Args:
            data (np.ndarray): Data points matrix.

        Returns:
            int: Optimal number of clusters.
        """
        max_clusters = self.default_params['max_clusters']
        inertias = []
        silhouette_scores = []

        print(f"Finding optimal number of clusters (max: {max_clusters})...")
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_scores[-1]:.4f}")

        kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        optimal_clusters = kl.elbow

        if optimal_clusters is None:
            optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
            print(f"Elbow method failed. Using highest silhouette score.")
        else:
            print(f"Elbow method found optimal clusters at {optimal_clusters}")

        print(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters

    def _evaluate_clustering(self, data, labels):
        """
        Evaluate clustering quality.

        Args:
            data (np.ndarray): Data points matrix.
            labels (np.ndarray): Cluster labels.

        Returns:
            None
        """

        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        print("\nClustering Evaluation Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")

    def _dimensionality_reduction_plot(self, data, params, n_components):
        """
        Create a dimensionality reduction plot.

        Args:
            data (list): [reduced_data, cluster_labels]
            params (dict): Plotting parameters.
            n_components (int): Number of dimensions (2 or 3).

        Returns:
            None
        """
        res, fclust = data
        
        n_clusters = self._find_optimal_clusters(res)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(res)

        self._evaluate_clustering(res, cluster_labels)

        fig = plt.figure(figsize=params['figsize'])
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(res[:, 0], res[:, 1], c=cluster_labels, alpha=params['alpha'], s=params['size'], cmap=params['cmap'])
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(res[:, 0], res[:, 1], res[:, 2], c=cluster_labels, alpha=params['alpha'], cmap=params['cmap'], s=params['size'])
            ax.set_zlabel(params['z_label'])
        
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        plt.title(f"{params['title']} (Optimal clusters: {n_clusters})")
        
        ticks = np.arange(n_clusters)
        cbar = plt.colorbar(scatter)
        cbar.set_ticks(ticks)
        
        print(f"Dimensionality reduction plot created with {n_components} components and {n_clusters} clusters")
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
        principalDF, explained_variance_ratio, fclust = data
        n_components = params.get('n_components', 2)
        
        print("\nPCA Analysis:")
        print(f"Number of components: {n_components}")
        print("Explained Variance Ratio:")
        for i, ratio in enumerate(explained_variance_ratio[:n_components]):
            print(f"  PC{i+1}: {ratio:.4f}")
        print(f"Total explained variance: {sum(explained_variance_ratio[:n_components]):.4f}")
        
        self._dimensionality_reduction_plot((principalDF, fclust), params, n_components)

    def _tsneplot(self, data, params):
        """
        Create a t-SNE plot.

        Args:
            data (list): [tsne_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        n_components = params.get('n_components', 2)
        print("\nt-SNE Plot:")
        print(f"Number of components: {n_components}")
        self._dimensionality_reduction_plot(data, params, n_components)

    def _umapplot(self, data, params):
        """
        Create a UMAP plot.

        Args:
            data (list): [umap_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        n_components = params.get('n_components', 2)
        print("\nUMAP Plot:")
        print(f"Number of components: {n_components}")
        self._dimensionality_reduction_plot(data, params, n_components)

    def _ivisplot(self, data, params):
        """
        Create an ivis plot.

        Args:
            data (list): [ivis_results, cluster_labels]
            params (dict): Plotting parameters.

        Returns:
            None
        """
        n_components = params.get('embedding_dims', 2)
        print("\nivis Plot:")
        print(f"Number of components: {n_components}")
        self._dimensionality_reduction_plot(data, params, n_components)
        
    def _spectralclusteringplot(self, data, params):
        """
        Create a comprehensive spectral clustering plot.

        Args:
            data (list): [X, n_clusters] where X is the data matrix and n_clusters is the number of clusters.
            params (dict): Plotting parameters.

        Returns:
            None
        """
        X, n_clusters = data
        print(f"\nPerforming Spectral Clustering with {n_clusters} clusters...")

        # Compute affinity matrix
        n_neighbors = min(15, X.shape[0] - 1)
        connectivity = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X).kneighbors_graph()
        affinity_matrix = 0.5 * (connectivity + connectivity.T)

        # Perform spectral clustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        cluster_labels = sc.fit_predict(affinity_matrix)

        # Compute Laplacian and its eigenvectors
        laplacian = csgraph.laplacian(affinity_matrix, normed=True)
        eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')

        # Create the main figure
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
        ax2.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
        ax2.set_title('Clustered Data')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')

        # Plot eigenvalues
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        ax3.set_title('Eigenvalues')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Eigenvalue')

        # Compute and plot silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_score(X, cluster_labels, metric='euclidean')
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax4.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
        ax4.set_title("Silhouette plot for the various clusters")
        ax4.set_xlabel("Silhouette coefficient values")
        ax4.set_ylabel("Cluster label")
        ax4.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax4.set_yticks([])  # Clear the yaxis labels / ticks

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

        # Compute and display metrics
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)

        metrics_text = (f"Silhouette Score: {silhouette_avg:.4f}\n"
                        f"Calinski-Harabasz Score: {calinski_harabasz:.4f}\n"
                        f"Davies-Bouldin Score: {davies_bouldin:.4f}")

        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        ax7.text(0.1, 0.7, metrics_text, fontsize=12, verticalalignment='top')

        plt.tight_layout()
        
        print("\nSpectral Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        self._save_and_show(params)

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
        if params['show']:
            plt.show()
        plt.close()