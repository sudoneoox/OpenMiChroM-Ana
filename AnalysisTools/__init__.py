
"""
OpenMiChroM-Ana: A package for analyzing chromosome structure data.

Required packages:
    - numpy
    - scipy
    - scikit-learn
    - joblib
    - numba
    - matplotlib
    - seaborn
    - umap-learn
    - kneed
    - pandas

For GPU support (optional):
    - cupy
    - cuml
    - cugraph
    - cudf

Installation:
    For CPU-only support:
    pip install numpy scipy scikit-learn joblib numba matplotlib seaborn umap-learn kneed pandas

    For GPU support (requires CUDA):
    pip install numpy scipy scikit-learn joblib numba matplotlib seaborn umap-learn kneed pandas cupy cuml cugraph cudf

Note: GPU support requires a CUDA-enabled GPU and appropriate CUDA toolkit installation.
"""


from .CompHelperCPU import ComputeHelpersCPU

try:
    from .CompHelperGPU import ComputeHelpersGPU
except ImportError:
    print("GPU support not available. Using CPU only.")
    ComputeHelpersGPU = None

from .PlotHelper import PlotHelper
from .Ana import Ana 
from .cndbToolsMini import cndbToolsMini


__all__ = ['Ana', 'ComputeHelpersCPU', 'ComputeHelpersGPU', 'PlotHelper', 'cndbToolsMini']