import numpy as np
import os
from OpenMiChroM.CndbTools import cndbTools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import util

class Ana:
    def __init__(self, base_folder):
        self.datasets = {}
        self.trajectories = {}
        self.results = {}
        self.base_folder = base_folder

    def add_dataset(self, label, folder, num_replicas):
        self.datasets[label] = {
            'folder': os.path.join(self.base_folder, folder),
            'num_replicas': num_replicas,
            'analysis_output': os.path.join(self.base_folder, folder, "analysis")
        }
        
    
        
    
    def HiCMapPlot(self, hic_sim, scale_sim, err_sim, outputPath, denseExperimental=None, denseSimulation=None, show=True, label=None,figsize=10):
        """ Expects 
            hic_sim, scale_sim, err_sim given from getHiCData
            @denseExpirimental: filepath to chromosome .dense file
            @denseSimulation: filepath to simulation probdist file
            @outputPath: filepath for output .png
            @show: True/False whether to show the HiC Map
            returns - .png of a HiC at the given outputPath (contact matrix)
        """
        if  denseSimulation != None:
            hic_sim, scale_sim, err_sim = util.getHiCData_simulation(denseSimulation)
        if denseExperimental != None:
            hic_exp, scale_exp, err_exp = util.getHiCData_experiment(denseExperimental, norm="first")
        
        comp = np.triu(hic_exp) + np.tril(hic_sim, k=1)
        plt.rcParams["figure.figsize"] = (figsize,figsize)
        mpl.rcParams['axes.linewidth'] = 2

        plt.matshow(comp, norm=mpl.colors.LogNorm(vmin=0.001, vmax=1.0), cmap='Reds')
        plt.title(f"Simulated vs. Expiremental Hi-C", pad=20)  
        plt.colorbar()  
        plt.savefig(outputPath)
        if show:
            plt.show()
    
    def GenomeDistancePlot(self, scale_exp, outputPath, denseExperimental=None, denseSimulation=None, show=True):
        """
        @scale_exp given as D from getHiCData function
        @denseExpirimental: filepath to chromosome .dense file
        @denseSimulation: filepath to simulation probdist file
        @outputPath: filepath for output .png
        @show: True/False whether to show the HiC Map
        returns - .png of Contact Probability and Genomic Distance plot
        """
        
        if denseSimulation != None:
            hic_sim, scale_sim, err_sim = util.getHiCData_simulation(denseSimulation)
        if denseExperimental != None:
            hic_exp, scale_exp, err_exp = util.getHiCData_experiment(denseExperimental, norm="first")
        
        mpl.rcParams['axes.linewidth'] = 2.
        cmap = sns.blend_palette(['white', 'red'], as_cmap=True)  
        cmap.set_bad(color='white')
        fig, ax = plt.subplots(figsize=(10, 10))  


        ax.loglog(range(len(scale_exp)), scale_exp, color='r', label='Exp.', linewidth=2)
        ax.loglog(range(len(scale_sim)), scale_sim, color='green', linestyle='--', label='Simulated', linewidth=2)
        ax.set_title('Scaling', loc='center')
        ax.set_xlabel('Genomic Distance')
        ax.set_ylabel('Probability')
        ax.set_xlim([1, len(scale_exp)])  
        ax.set_ylim([min(scale_exp), max(scale_exp)])  
        ax.legend()
        
        plt.savefig(outputPath)
        if show:
            plt.show()
    
    def ErrorPlot(self, folderPath, outputPath, filename):
       return
        

         
        
        
        
                

    def process_trajectories(self, label, filename):
        config = self.datasets[label]
        trajs_xyz = []
        
        for i in range(1, config['num_replicas'] + 1):
            traj = self.load_and_process_trajectory(
                config['folder'], i, filename=filename
            )
            if traj.size > 0:
                trajs_xyz.append(traj)
        
        if trajs_xyz:
            self.trajectories[label] = np.vstack(trajs_xyz)
            print(f'Trajectory for {label} has shape {self.trajectories[label].shape}')
        else:
            print(f"No valid trajectories found for {label}")

    def load_and_process_trajectory(self, folder, replica, filename, key=None):
        path = os.path.join(folder, f'iteration_{replica}/{filename}')
        
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return np.array([])
        else:
            print(f"Processing file: {path}")

        try:
            trajectory = cndbTools.load(filename=path)            
            list_traj = [int(k) for k in trajectory.cndb.keys() if k != 'types']
            list_traj.sort()
            beadSelection = trajectory.dictChromSeq[key] if key else None
            first_snapshot, last_snapshot = list_traj[0], list_traj[-1]
            trajs_xyz = cndbTools.xyz(frames=[first_snapshot, last_snapshot+1, 2000], XYZ=[0,1,2], beadSelection=beadSelection)
            return trajs_xyz
        
        except Exception as e:
            print(f"Error processing trajectory {replica}: {str(e)}")
            return np.array([])

    def perform_pca(self, label, n_components=2):
        """Perform PCA on the data to reduce dimensions."""
        if label not in self.trajectories:
            print(f"No data available for {label}.")
            return None
        
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(self.trajectories[label].reshape(self.trajectories[label].shape[0], -1))
        print(f"PCA completed for {label}. Explained variance ratio: {pca.explained_variance_ratio_}")
        return transformed_data

    def cluster_data(self, data, n_clusters=3):
        """Cluster the data using k-means."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        print("Clustering completed.")
        return kmeans.labels_, kmeans.cluster_centers_
    
    def plot_clusters(self, data, labels, title='Cluster Plot'):
        """Plot the clustered data."""
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
