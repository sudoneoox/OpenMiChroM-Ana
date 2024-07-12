import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from numba import jit
from joblib import Parallel, delayed, Memory

memory = Memory(location='.', verbose=0)

CUDA_AVAILABLE = False

def set_cuda_availability(available):
    global CUDA_AVAILABLE
    CUDA_AVAILABLE = available

def set_cuda_availability(available):
    global CUDA_AVAILABLE
    CUDA_AVAILABLE = available



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
    
def calc_distance_gpu(X: np.array, metric: str) -> np.array:
    # This function will only be called if CUDA is available
    import cupy as cp
    from cupyx.scipy import stats as cp_stats

    X_gpu = cp.asarray(X)
    
    if metric == 'euclidean':
        result = cp.sqrt(cp.sum((X_gpu[:, None] - X_gpu) ** 2, axis=2))
    elif metric == 'pearson':
        mean = cp.mean(X_gpu, axis=1, keepdims=True)
        std = cp.std(X_gpu, axis=1, keepdims=True)
        normalized = (X_gpu - mean) / std
        result = 1 - cp.dot(normalized, normalized.T) / X_gpu.shape[1]
    elif metric == 'spearman':
        rank_data = cp_stats.rankdata(X_gpu, axis=1)
        mean = cp.mean(rank_data, axis=1, keepdims=True)
        std = cp.std(rank_data, axis=1, keepdims=True)
        normalized = (rank_data - mean) / std
        result = 1 - cp.dot(normalized, normalized.T) / rank_data.shape[1]
    elif metric == 'contact':
        result = cp.sum(cp.abs(X_gpu[:, None] - X_gpu), axis=2)
    elif metric == 'log2_contact':
        log2_X_gpu = cp.log2(X_gpu + 1)
        result = cp.sum(cp.abs(log2_X_gpu[:, None] - log2_X_gpu), axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return cp.asnumpy(result)


    
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
        return calc_distance_gpu(X, metric)
    else:
        return Parallel(n_jobs=n_jobs)(delayed(calc_dist)(val, metric) for val in trajectories)
