from setuptools import setup, find_packages

setup(
    name="OpenMiChroM-Ana",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'joblib',
        'numba',
        'matplotlib',
        'seaborn',
        'umap-learn',
        'kneed',
        'pandas',
    ],
    extras_require={
        'gpu': [
            'cupy',
            'cuml',
            'cugraph',
            'cudf',
        ],
    },
    author="Diego Coronado",
    author_email="diegoa2992@gmail.com",
    description="A package for analyzing chromosome structure data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sudoneoox/OpenMiChroM-Ana",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)