import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from AnalysisTools.CompHelperCPU import ComputeHelpersCPU
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
        self.compute_helpers.setMem(memory_location=self.cacheStorage)
        
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

    def _distmap(self, data, params):
        """
        Create a distance map.
        Args:
        data (list): [distance_matrix] containing distances.
        params (dict): Plotting parameters.
        Returns:
        None
        """
        try:
            n_maps = data.shape[0]
            rows = int(np.ceil(np.sqrt(n_maps)))
            cols = int(np.ceil(n_maps / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(params['size'] *cols, 4*rows))
            axes = axes.flatten()
            
            for i in range(n_maps):
                im = axes[i].imshow(data[i], cmap='Spectral_r')
                axes[i].set_title(f'Frame {i+1}')
                axes[i].set_xlabel('Beads')
                axes[i].set_ylabel('Beads')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
            
            for i in range(n_maps, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            norm_str = f"_{params['norm']}" if params['norm'] else "None"
            method_str = f"_{params['method']}" if params['method'] else "None"
            plt.savefig(os.path.join(params['outPath'], f'{params['label']}_dist_maps_{params['metric']}{norm_str}{method_str}.png'))
            plt.show()
            plt.close()
        except Exception as e:
            print(f"An error occurred while plotting: {str(e)}")


    def _dimensionality_reduction_plot(self, data, params):
        result, additional_info, _ = data
        n_components = params.get('n_components', 2)
        plot_type = params.get("plot_type")
        labels = params.get('labels', [])
        n_clusters = params.get('n_clusters', 2)
        clustering_method = params.get('cluster_method', 'kmeans')
        clustering_params = params.get('clustering_params', {})
        scatter_plots = []        

        fig = plt.figure(figsize=params['figsize'])

        # main scatter plot
        ax_main = fig.add_subplot(111)
        
        if params.get('show_distribution', False):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_main)
            ax_top = divider.append_axes("top", size="20%", pad=0.1)
            ax_right = divider.append_axes("right", size="20%", pad=0.1)

        # Use a fixed colormap for the clusters
        color_palette = sns.color_palette(params.get('color_palette', 'bright'), n_colors=n_clusters)

        if clustering_method in self.compute_helpers.getClusteringMethods():
            cluster_labels, clustering_info = self.compute_helpers.run_clustering(
                method=clustering_method,
                X=result,
                n_clusters=n_clusters,
                n_components=n_components,
                **clustering_params
            )
        else:
            raise ValueError(f"Invalid clustering method: {clustering_method}. Avaliable clustering methods are: {self.compute_helpers.getClusteringMethods()}")

        # Perform clustering and evaluation
        _, evaluation_metrics = self.compute_helpers.evaluate_clustering(result, cluster_labels)
                
        if n_components == 1 or n_components == 2:
            for i, label in enumerate(labels):
                cluster_data = result[cluster_labels == i]
                scatter = ax_main.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        c=[color_palette[i]], label=label,
                        alpha=params['alpha'], s=params['size'])
            scatter_plots.append(scatter)
        elif n_components == 3:
            ax_main = fig.add_subplot(111, projection='3d')
            
            scatter = ax_main.scatter(result[:, 0], result[:, 1], result[:, 2],
                                c=cluster_labels,
                                alpha=params['alpha'],
                                s=params['size'],
                                cmap='Set1')  # Use a discrete colormap
            ax_main.set_xlabel(params['x_label'])
            ax_main.set_ylabel(params['y_label'])
            ax_main.set_zlabel(params['z_label'])
            scatter_plots.append(scatter)
    
        ax_main.set_xlabel(params['x_label'], fontsize=14)
        ax_main.set_ylabel(params['y_label'], fontsize=14)
    
 
        if params.get('show_distribution', False):
            for i, label in enumerate(labels):
                cluster_data = result[cluster_labels == i]
                sns.kdeplot(x=cluster_data[:, 0], ax=ax_top, fill=True, color=color_palette[i], label=label)
                sns.kdeplot(y=cluster_data[:, 1], ax=ax_right, fill=True, color=color_palette[i])
            
            
            dist_y_scale = params.get('dist_y_scale', 1.0)
            ax_top.set_ylim(0, ax_top.get_ylim()[1] * dist_y_scale)
            ax_right.set_xlim(0, ax_right.get_xlim()[1] * dist_y_scale)

            ax_top.set_xlim(ax_main.get_xlim())
            ax_right.set_ylim(ax_main.get_ylim())
            
            # Remove ticks, labels, and spines from distribution plots
            for ax in [ax_top, ax_right]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
            fig.set_size_inches(params['figsize'][0], params['figsize'][1] * (1 + 0.2 * (1/dist_y_scale * 1.2)))


            # Align the limits of distribution plots with the main plot
            ax_top.set_xlim(ax_main.get_xlim())
            ax_right.set_ylim(ax_main.get_ylim())

        legend = ax_main.legend(title="Cell Types", loc='upper left', fontsize=12, title_fontsize=12, frameon=True)
        legend.get_frame().set_alpha(0.6)
                
        if params.get('show_legend_seperate', False):
            ax_main.get_legend().remove()
            
            figLegend = plt.figure(figsize=(6, 7))
            ax_legend = figLegend.add_subplot(111)
            
            # Create new patches for the legend instead of reusing scatter plots
            legend_elements = [plt.scatter([], [], c=[color_palette[i]], label=label, 
                                        s=params['size'], alpha=params['alpha']) 
                            for i, label in enumerate(labels)]
            
            legend = ax_legend.legend(handles=legend_elements, title="Cell Types", 
                                    loc='center', fontsize=12, title_fontsize=12, frameon=True)
            legend.get_frame().set_alpha(0.6)
            
            ax_legend.axis('off')
            
            legend_path = params.get('outputFileName').rsplit('.', 1)[0] + '_legend.png'
            figLegend.savefig(legend_path, bbox_inches='tight', transparent=False)
            
            plt.close(figLegend)
                
            
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        fig.suptitle(params['title'])

        
        plt.tight_layout()
        
        plt.figure(fig.number)
        plt.savefig(params['outputFileName'], bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # Prepare additional info text
        additional_info_text = ""
        if plot_type == 'pcaplot':
            if isinstance(additional_info, np.ndarray) and additional_info.size == n_components:
                additional_info_text = "Explained variance:\n" + "\n".join([f"PC{i+1}: {var:.2%}" for i, var in enumerate(additional_info)])
        elif plot_type == 'tsneplot':
            if isinstance(additional_info, float):
                additional_info_text = f"KL divergence: {additional_info:.4f}"
        elif plot_type == 'umapplot':
            if isinstance(params.get('embedding'), np.ndarray):
                embedding = params.get('embedding')
                singular_values = embedding.flatten()[:5]
                additional_info_text = "UMAP Embedding:\n" + "\n".join([f"{val:.2f}" for val in singular_values])
        elif plot_type == 'svdplot':
            if isinstance(additional_info, np.ndarray):
                additional_info_text = "Top singular values:\n" + "\n".join([f"{val:.2f}" for val in additional_info[:5]])
        elif plot_type == 'mdsplot':
            if isinstance(additional_info, float):
                additional_info_text = f"Stress: {additional_info:.4f}"
        
        # Add n_components_95 information if available
        n_components_95 = params.get('n_components_95')
        if n_components_95 is not None:
            additional_info_text += f"\n\nComponents for 95% variance: {n_components_95}"
        
        # print additional info
        # print(additional_info)

        print(f"Dimensionality reduction plot created with {n_components} components")
        
        # Prepare logging information
        log_info = f"""
            {'='*50}
            File: {params.get('outputFileName')}
            {'='*50}
            Parameters:
            - Plot Type: {plot_type}
            - n_components: {n_components}
            - n_clusters: {params.get('n_clusters')}
            - cluster_method: {params.get('cluster_method')}
            - method: {params.get('method')}
            - metric: {params.get('metric')}
            - norm: {params.get('norm')}

            Results:
            Clustering Metrics:
            - Silhouette Score: {evaluation_metrics[0]:.4f} (higher is better, range: [-1, 1])
            - Calinski-Harabasz Index: {evaluation_metrics[1]:.4f} (higher is better)
            - Davies-Bouldin Index: {evaluation_metrics[2]:.4f} (lower is better)

            {'='*50}
            """
        
        # Write to log file
        log_path = params.get('logPath')
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
        
    def fetch_params(self, outPath, func_type, func_name, args, method, metric, norm, sample_size, n_clusters, n_components, labels=None, extra_params=None, user_params=None):
        cluster_method_save = 'kmeans' if extra_params['cluster_method'] == 'kmeans' else extra_params['cluster_method']
        
        base_params = {
            'outputFileName': os.path.join(outPath, f'{func_name.upper()}/{func_name}_plot_{args}_{method}_{metric}_{norm}_{cluster_method_save}.png'),
            'logPath': os.path.join(outPath, f'{func_name.upper()}/{func_name}_{args}_{cluster_method_save}_log.txt'),
            'plot_type': f'{func_name}plot',
            'cmap': 'viridis',
            'title': f'{func_name.upper()} of {args}',
            'method': method,
            'metric': metric,
            'norm': norm,
            'sample_size': sample_size,
            'n_clusters': n_clusters,
            'n_components': n_components,
            'size': 50,
            'alpha': 0.7,
            'labels': labels if labels is not None else args,
        }
        
        if func_type == 'reduction':
            base_params.update({
                'x_label': f'{func_name.upper()} 1',
                'y_label': f'{func_name.upper()} 2',
                'z_label': f'{func_name.upper()} 3' if n_components > 2 else None,
                'figsize': (12, 8),
                'show_distribution': True,
                'dist_y_scale': 2,
                'show_legend_seperate': True,
                'color_palette':'bright',
                'clustering_method':'kmeans',
                'clustering_params': {'randomState':47}
            })
        elif func_type == 'clustering':
            base_params.update({
                'x_label': 'Feature 1',
                'y_label': 'Feature 2',
            })
        
        if extra_params:
            base_params.update(extra_params)
        
        if user_params:
            base_params.update(user_params)
        
        return base_params
            
    """======================================================== Getters / Setters ========================================================"""
    def getInitialParams(self):
        return self.default_params

    
    def setInitialParams(self, params: dict):
        for key in params:
            self.default_params[key] = params[key]
        print(f'the updated initial plot parameters are: {self.default_params}')
        
    def setMeMForComputeHelpers(self, memory_location: str):
        self.compute_helpers.setMem(memory_location=memory_location)