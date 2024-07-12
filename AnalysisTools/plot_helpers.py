import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D

class PlotHelper:
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
            'fraction': 0.046
        }

    def plot(self, plot_type, data, plot_params=None, **kwargs):
        if plot_params:
            params = {**self.default_params, **plot_params}
        else:
            params = self.default_params.copy()
        
        plot_func = getattr(self, f"_{plot_type.lower()}", None)
        if plot_func:
            plot_func(data, params, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _hicmapplot(self, data, params):
        hic_exp, hic_sim = data
        comp = np.triu(hic_exp) + np.tril(hic_sim, k=1)
        plt.figure(figsize=params['figsize'])
        plt.matshow(comp, norm=plt.colors.LogNorm(vmin=params['vmin'], vmax=params['vmax']), cmap=params['cmap'])
        plt.title(params['title'])
        plt.colorbar()
        self._save_and_show(params)

    def _genomedistanceplot(self, data, params):
        scale_exp, scale_sim = data
        fig, ax = plt.subplots(figsize=params['figsize'])
        ax.loglog(range(len(scale_exp)), scale_exp, color=params['colors'][0], label=params['labels'][0], linestyle=params['linestyles'][0], linewidth=params['linewidths'][0])
        ax.loglog(range(len(scale_sim)), scale_sim, color=params['colors'][1], linestyle=params['linestyles'][1], label=params['labels'][1], linewidth=params['linewidths'][1])
        ax.set_title(params['title'])
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        ax.legend()
        self._save_and_show(params)

    def _errorplot(self, data, params):
        iterations = np.arange(len(data))
        plt.figure(figsize=params['figsize'])
        plt.plot(iterations, data, marker=params['marker'], linestyle=params['linestyle'], color=params['color'])
        plt.xlabel(params['x_label'])
        plt.ylabel(params['y_label'])
        plt.title(params['title'])
        plt.grid(True)
        plt.legend()
        self._save_and_show(params)

    def _dendrogram(self, data, params):
        plt.figure(figsize=params['figsize'])
        hierarchy.dendrogram(data)
        plt.title(params['title'])
        self._save_and_show(params)

    def _euclidiandistmap(self, data, params):
        fig, ax = plt.subplots(1, 1, figsize=params['figsize'])
        p = ax.imshow(data[0], vmin=params['vmin'], vmax=params['vmax'], cmap=params['cmap'])
        plt.colorbar(p, ax=ax, fraction=params['fraction'])
        plt.title(params['title'])
        plt.xlabel(params['x_label'])
        plt.ylabel(params['y_label'])
        self._save_and_show(params)

    def _dimensionality_reduction_plot(self, data, params, n_components):
        res, fclust = data
        fig = plt.figure(figsize=params['figsize'])
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(res[:, 0], res[:, 1], c=fclust, alpha=params['alpha'], s=params['size'], cmap=params['cmap'])
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(res[:, 0], res[:, 1], res[:, 2], c=fclust, alpha=params['alpha'], cmap=params['cmap'], s=params['size'])
            ax.set_zlabel(params['z_label'])
        
        ax.set_xlabel(params['x_label'])
        ax.set_ylabel(params['y_label'])
        plt.title(params['title'])
        
        ticks = np.arange(1, len(set(fclust)) + 1)
        cbar = plt.colorbar(scatter)
        cbar.set_ticks(ticks)
        
        self._save_and_show(params)

    def _pcaplot(self, data, params):
        principalDF, fclust, explained_variance_ratio = data
        n_components = params.get('n_components', 2)
        self._dimensionality_reduction_plot((principalDF.values, fclust), params, n_components)

    def _tsneplot(self, data, params):
        n_components = params.get('n_components', 2)
        self._dimensionality_reduction_plot(data, params, n_components)

    def _umapplot(self, data, params):
        n_components = params.get('n_components', 2)
        self._dimensionality_reduction_plot(data, params, n_components)

    def _ivisplot(self, data, params):
        n_components = params.get('embedding_dims', 2)
        self._dimensionality_reduction_plot(data, params, n_components)

    def _save_and_show(self, params):
        plt.savefig(params['outputFileName'])
        if params['show']:
            plt.show()
        plt.close()

# Usage
plot_helper = PlotHelper()
plot_helper.plot('hicmapplot', data, plot_params={'title': 'HiC Map', 'cmap': 'coolwarm'})