import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from numba import jit, cuda
from joblib import Parallel, delayed, Memory
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from kneed import KneeLocator
from sklearn.cluster import KMeans


memory = Memory(location='.', verbose=0)

CUDA_AVAILABLE = False

def set_cuda_availability(available):
    global CUDA_AVAILABLE
    CUDA_AVAILABLE = available
    if CUDA_AVAILABLE == True:
        import cupy as cp



def getHiCData_simulation(filepath):
    """
    Returns: 
        r: HiC Data
        D: Scaling
        err: error data 
    """
    contactMap = np.loadtxt(filepath)
    r=np.triu(contactMap, k=1) 
    r = normalize(r, axis=1, norm='max') 
    rd = np.transpose(r) 
    r=r+rd + np.diag(np.ones(len(r))) 

    D1=[]
    err = []
    for i in range(0,np.shape(r)[0]): 
        D1.append((np.mean(np.diag(r,k=i)))) 
        err.append((np.std(np.diag(r,k=i))))
    
    return(r,D1,err)
    
def getHiCData_experiment(filepath, cutoff=0.0, norm="max"):
    """
    Returns: 
        r: HiC Data
        D: Scaling
        err: error data 
    """
    contactMap = np.loadtxt(filepath)
    r = np.triu(contactMap, k=1)
    r[np.isnan(r)]= 0.0
    r = normalize(r, axis=1, norm="max")
    
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
    r=r+rd + np.diag(np.ones(len(r)))

    D1=[]
    err = []
    for i in range(0,np.shape(r)[0]): 
        D1.append((np.mean(np.diag(r,k=i)))) 
        err.append((np.std(np.diag(r,k=i))))
    D=np.array(D1)#/np.max(D1)
    err = np.array(err)

    return(r,D,err)


def pad_array(array, target_shape):
    """
    Pads a 3D array to the target shape with zeros.
    
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

def calc_dist(X: np.array, metric: str) -> np.array:
    metricPtrs = {
        'euclidean': lambda x: squareform(pdist(x, metric='euclidean')),
        'pearsons': pearson_distance,
        'spearman': spearman_distance,
        'contact': lambda x: squareform(pdist(x, metric='cityblock')),
        "log2_contact": log2_contact_distance
    }
    try:
        return metricPtrs[metric](X)
    except KeyError:
        raise KeyError(f"Invalid metric: {metric}. Available metrics are: {list(metricPtrs.keys())}")
    


@cuda.jit
def gpu_distance(traj, dist, metric):
    i, j = cuda.grid(2)
    if i < traj.shape[0] and j < traj.shape[0]:
        if metric == 'euclidean':
            dist[i, j] = gpu_euclidean_distance(traj[i], traj[j])
        elif metric == 'pearson':
            dist[i, j] = gpu_pearson_distance(traj[i], traj[j])
        elif metric == 'spearman':
            dist[i, j] = gpu_spearman_distance(traj[i], traj[j])
        elif metric == 'contact':
            dist[i, j] = gpu_contact_distance(traj[i], traj[j])
        elif metric == 'log2_contact':
            dist[i, j] = gpu_log2_contact_distance(traj[i], traj[j])
            
def calc_distance_gpu(traj, metric):
    traj = np.array(traj)
    dist = np.zeros((traj.shape[0], traj.shape[0]), dtype=np.float32)
    threadsperblock = (16, 16)
    blockspergrid_x = (traj.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (traj.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    gpu_distance[blockspergrid, threadsperblock](traj, dist, metric)
    return dist
    
def pearson_distance(X: np.array) -> np.array:
    """
    Calculate the Pearson correlation distance matrix.

    Args:
        X (np.array): The data matrix.

    Returns:
        np.array: The Pearson correlation distance matrix.
    """
    corr = np.corrcoef(X)
    return 1 - corr


def spearman_distance(X: np.array) -> np.array:
    """
    Calculate the Spearman correlation distance matrix.

    Args:
        X (np.array): The data matrix.

    Returns:
        np.array: The Spearman correlation distance matrix.
    """
    rank_data = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, X)
    return 1 - np.corrcoef(rank_data)

def log2_contact_distance(X: np.array) -> np.array:
    """
    Calculate the log2 contact count distance matrix.

    Args:
        X (np.array): The data matrix.

    Returns:
        np.array: The log2 contact count distance matrix.
    """
    log2_X = np.log2(X + 1)
    return squareform(pdist(log2_X, metric='cityblock'))



def norm_distMatrix(matrix: np.array, norm: str):
    normPtrs = {
        'ice': ice_normalization,
        'log_transform': lambda m: np.log2(m + 1),
        'vc': lambda m: m / m.sum(axis=1, keepdims=True),
        'kr': kr_norm,
    }
    try:
        return normPtrs[norm](matrix)
    except KeyError:
        raise KeyError(f"Invalid normalization method: {norm}. Available methods are: {list(normPtrs.keys())}")
    
@cuda.jit(device=True)
def gpu_euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

@cuda.jit(device=True)
def gpu_pearson_distance(a, b):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.sqrt(np.sum((a - mean_a)**2) / (len(a) - 1))
    std_b = np.sqrt(np.sum((b - mean_b)**2) / (len(b) - 1))
    cov = np.sum((a - mean_a) * (b - mean_b)) / (len(a) - 1)
    return 1 - (cov / (std_a * std_b))

@cuda.jit(device=True)
def gpu_spearman_distance(a, b):
    def rank(x):
        temp = sorted(enumerate(x), key=lambda x: x[1])
        ranks = [0] * len(x)
        for i, (index, value) in enumerate(temp):
            ranks[index] = i
        return ranks
    
    rank_a = rank(a)
    rank_b = rank(b)
    return gpu_pearson_distance(rank_a, rank_b)

@cuda.jit(device=True)
def gpu_contact_distance(a, b):
    return np.sum(np.abs(a - b))

@cuda.jit(device=True)
def gpu_log2_contact_distance(a, b):
    log2_a = np.log2(a + 1)
    log2_b = np.log2(b + 1)
    return np.sum(np.abs(log2_a - log2_b))


@jit(nopython=True)
def ice_normalization(matrix: np.array, max_iter: int=100, tolerance: float=1e-5) -> np.array:
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
def kr_norm(matrix: np.array, max_iter: int=100, tolerance: float=1e-5) -> np.array:
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



@memory.cache
def cached_calc_dist(trajectories, metric, execution_mode, n_jobs):
    if execution_mode == 'cuda' and CUDA_AVAILABLE:
        return [calc_distance_gpu(traj, metric) for traj in trajectories]
    else:
        return Parallel(n_jobs=n_jobs)(delayed(calc_dist)(val, metric) for val in trajectories)


def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method and silhouette score.
    """
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Elbow method
    kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
    elbow = kl.elbow if kl.elbow else max_clusters
    
    # Silhouette method
    silhouette_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Return the smaller of the two to be conservative
    return min(elbow, silhouette_optimal)

def evaluate_clustering(data, labels):
    """
    Evaluate clustering quality.

    Args:
        data (np.ndarray): Data points matrix.
        labels (np.ndarray): Cluster labels.

    Returns:
        [silhoutte, calinski_harabasz, davies_bouldin] (float): scores
    """

    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    
    print("\nClustering Evaluation Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
    print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    return silhouette, calinski_harabasz, davies_bouldin

    