# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 13 January 2025

import os
from pathlib import Path
import urllib.request

from datetime import datetime
import gc
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pyriemann.classification import SVC
from pyriemann.estimation import (
    BlockCovariances, 
    Covariances,
    Kernels,
    Shrinkage,
)
ker_est_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]
from pyriemann.utils.covariance import cov_est_functions

# ------------------------------------------------------------
# define constant values
block_size = 4 # number of channels for HbO and HbR
n_jobs = -1 # use all available cores
cv_splits = 5 # number of cross-validation folds
random_state = 42 # random state for reproducibility
n_clusters = 5 # number of clusters for k-means
n_init = 100 # number of initializations
max_iter = 5 # maximum number of iterations
upsampling_freq = 5 # frequency to which the data have been upsampled

# ------------------------------------------------------------
# define custom classes and functions
class WindowTransformer(BaseEstimator, TransformerMixin):
    """Splits the time series into non-overlapping windows of shape (n_channels, window_size). 
        Author: Luca Naudszus."""
    def __init__(self, window_size=upsampling_freq*10, step_size=None):
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"
        n_channels = X.shape[0]
        assert X.ndim == 2 and (n_channels in {8, 16}), "Unexpected input shape"
        n_samples = X.shape[1]
        if self.step_size == None: 
            n_windows = n_samples // self.window_size
            truncated_length = n_windows * self.window_size
            X_windows = X[:, :truncated_length].reshape(n_channels, n_windows, self.window_size)
            return np.transpose(X_windows, (1, 0, 2))
        else: 
            n_windows = (n_samples - self.window_size) // self.step_size + 1
            # Extract windows using a sliding approach
            X_windows = np.lib.stride_tricks.sliding_window_view(X, (n_channels, self.window_size), axis=1)[:, :, ::self.step_size]
            return np.transpose(X_windows, (2, 0, 1))
        
    def transform_labels(self, y):
        """Adjust labels to match the windowed data."""
        # Convert y to np.ndarray if it's a numpy string type
        if isinstance(y, np.ndarray) and y.dtype == np.str_:
            y = y.astype(str)  # Convert to a string array if needed

        # Ensure y is a numpy array (converting from list if necessary)
        if isinstance(y, list):  
            y = np.array(y)

        # Handle the conversion of labels into the expected format (e.g., numeric)
        if y.dtype.kind in 'SU':  # If y contains strings, map to numeric or categories
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        if self.step_size is None:
            n_windows = len(y) // self.window_size
            return y[: n_windows * self.window_size : self.window_size]
        else:
            return y[: len(y) - self.window_size + 1 : self.step_size]
        
class ListTimeSeriesWindowTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list of time series into non-overlapping windows of shape (n_channels, window_size). 
        Author: Luca Naudszus."""
    def __init__(self, window_size = upsampling_freq*10, step_size=None):
        self.window_size = window_size
        self.step_size = step_size
        self.base_transformer = WindowTransformer(window_size, step_size)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, list), "Input must be a list of NumPy arrays"
        transformed_X = [self.base_transformer.transform(x) for x in X]
        return np.concatenate(transformed_X, axis=0)
        
    def fit_transform(self, X, y=None):
        """Ensures y is transformed along with X."""
        transformed_X = self.transform(X)
        
        if y is not None:
            transformed_y = [self.transform_labels(yi) for yi in y]
            transformed_y = np.concatenate(transformed_y, axis=0)
            return transformed_X, transformed_y

        return transformed_X

class Stacker(TransformerMixin):
    """Stacks values of a DataFrame column into a 3D array."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "Input must be a DataFrame"
        assert X.shape[1] == 1, "DataFrame must have only one column"
        return np.stack(X.iloc[:, 0].values)


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flattens the last two dimensions of an array.
        ColumnTransformer requires 2D output, so this transformer
        is needed"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)


class HybridBlocks(BaseEstimator, TransformerMixin):
    """Estimation of block kernel or covariance matrices with
    customizable metrics and shrinkage.

    Perform block matrix estimation for each given time series,
    computing either kernel matrices or covariance matrices for
    each block based on the specified metrics. The block matrices
    are then concatenated to form the final block diagonal matrices.
    It is possible to add different shrinkage values for each block.
    This estimator is helpful when dealing with data from multiple
    sources or modalities (e.g. fNIRS, EMG, EEG, gyro, acceleration),
    where each block corresponds to a different source or modality
    and benefits from separate processing and tuning.

    Parameters
    ----------
    block_size : int | list of int
        Sizes of individual blocks given as int for same-size blocks,
        or list for varying block sizes.
    metrics : string | list of string, default='linear'
        The metric(s) to use when computing matrices between channels.
        For kernel matrices, supported metrics are those from
        ``pairwise_kernels``: 'linear', 'poly', 'polynomial',
        'rbf', 'laplacian', 'cosine', etc.
        For covariance matrices, supported estimators are those from
        pyRiemann: 'scm', 'lwf', 'oas', 'mcd', etc.
        If a list is provided, it must match the number of blocks.
    shrinkage : float | list of float, default=0
        Shrinkage parameter(s) to regularize each block's matrix.
        If a single float is provided, it is applied to all blocks.
        If a list is provided, it must match the number of blocks.
    n_jobs : int, default=None
        The number of jobs to use for the computation.
    **kwargs : dict
        Any further parameters are passed directly to the kernel function(s)
        or covariance estimator(s).

    See Also
    --------
    BlockCovariances
    """

    def __init__(self, block_size, metrics="linear", shrinkage=0,
                 n_jobs=None, **kwargs):
        self.block_size = block_size
        self.metrics = metrics
        self.shrinkage = shrinkage
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fit.

        Prepare per-block transformers.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time series.
        y : None
            Not used, here for compatibility with scikit-learn API.

        Returns
        -------
        self : HybridBlocks instance
            The HybridBlocks instance.
        """
        n_matrices, n_channels, n_times = X.shape

        # Determine block sizes
        if isinstance(self.block_size, int):
            num_blocks = n_channels // self.block_size
            remainder = n_channels % self.block_size
            self.blocks = [self.block_size] * num_blocks
            if remainder > 0:
                self.blocks.append(remainder)
        elif isinstance(self.block_size, list):
            self.blocks = self.block_size
            if sum(self.blocks) != n_channels:
                raise ValueError(
                    "Sum of block sizes mustequal number of channels"
                    )
        else:
            raise ValueError(
                "block_size must be int or list of ints"
                )

        # Compute block indices
        self.block_indices = []
        start = 0
        for size in self.blocks:
            end = start + size
            self.block_indices.append((start, end))
            start = end

        # Handle metrics parameter
        n_blocks = len(self.blocks)
        if isinstance(self.metrics, str):
            self.metrics_list = [self.metrics] * n_blocks
        elif isinstance(self.metrics, list):
            if len(self.metrics) != n_blocks:
                raise ValueError(
                    f"Length of metrics list ({len(self.metrics)}) "
                    f"must match number of blocks ({n_blocks})"
                )
            self.metrics_list = self.metrics
        else:
            raise ValueError(
                "Parameter 'metrics' must be a string or a list of strings."
            )

        # Handle shrinkage parameter
        if isinstance(self.shrinkage, (float, int)):
            self.shrinkages = [self.shrinkage] * n_blocks
        elif isinstance(self.shrinkage, list):
            if len(self.shrinkage) != n_blocks:
                raise ValueError(
                    f"Length of shrinkage list ({len(self.shrinkage)}) "
                    f"must match number of blocks ({n_blocks})"
                )
            self.shrinkages = self.shrinkage
        else:
            raise ValueError(
                "Parameter 'shrinkage' must be a float or a list of floats."
            )

        # Build per-block pipelines
        self.block_names = [f"block_{i}" for i in range(n_blocks)]

        transformers = []
        for i, (indices, metric, shrinkage_value) in enumerate(
                zip(self.block_indices, self.metrics_list, self.shrinkages)):
            block_name = self.block_names[i]

            # Build the pipeline for this block
            block_pipeline = make_pipeline(
                Stacker(),
            )

            # Determine if the metric is a kernel or a covariance estimator
            if metric in ker_est_functions:
                # Use Kernels transformer
                estimator = Kernels(
                    metric=metric,
                    n_jobs=self.n_jobs,
                    **self.kwargs
                )
                block_pipeline.steps.append(('kernels', estimator))
            elif metric in cov_est_functions.keys():
                # Use Covariances transformer
                estimator = Covariances(
                    estimator=metric,
                    **self.kwargs
                )
                block_pipeline.steps.append(('covariances', estimator))
            else:
                raise ValueError(
                    f"Metric '{metric}' is not recognized as a kernel "
                    f"metric or a covariance estimator."
                )

            # add shrinkage if provided
            # TODO: add support for different shrinkage types at some point?
            if shrinkage_value != 0:
                shrinkage_transformer = Shrinkage(shrinkage=shrinkage_value)
                block_pipeline.steps.append(
                    ('shrinkage', shrinkage_transformer)
                    )

            # Add the flattening transformer at the end of the pipeline
            block_pipeline.steps.append(('flatten', FlattenTransformer()))

            transformers.append((block_name, block_pipeline, [block_name]))

        # create the columncransformer with per-block pipelines
        self.preprocessor = ColumnTransformer(transformers)

        # Prepare the DataFrame
        X_df = self._prepare_dataframe(X)

        # Fit the preprocessor
        self.preprocessor.fit(X_df)

        return self

    def transform(self, X):
        """Estimate block kernel or covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time series.

        Returns
        -------
        M : ndarray, shape (n_matrices, n_channels, n_channels)
            Block diagonal matrices (kernel or covariance matrices).
        """
        check_is_fitted(self, 'preprocessor')

        # make the df wheren each block is 1 column
        X_df = self._prepare_dataframe(X)

        # Transform the data
        transformed_blocks = []
        data_transformed = self.preprocessor.transform(X_df)

        # calculate the number of features per block
        features_per_block = [size * size for size in self.blocks]

        # compute the indices where to split the data
        split_indices = np.cumsum(features_per_block)[:-1]

        # split the data into flattened blocks
        blocks_flat = np.split(data_transformed, split_indices, axis=1)

        # reshape each block back to its original shape
        for i, block_flat in enumerate(blocks_flat):
            size = self.blocks[i]
            block = block_flat.reshape(-1, size, size)
            transformed_blocks.append(block)

        # Construct the block diagonal matrices using scipy
        M_matrices = np.array([
            block_diag(*[Xt[i] for Xt in transformed_blocks])
            for i in range(X.shape[0])
        ])
        return M_matrices

    def _prepare_dataframe(self, X):
        """Converts the data into a df with eac hblock as column."""
        data_dict = {}
        for i, (start, end) in enumerate(self.block_indices):
            data_dict[self.block_names[i]] = list(X[:, start:end, :])
        return pd.DataFrame(data_dict)

# ------------------------------------------------------------
### Load data
# Load the dataset
npz = np.load('./data/ts_two_blocks.npz')
X = []
for array in list(npz.files):
    X.append(npz[array])
doc = pd.read_csv('./data/doc_two_blocks.csv', index_col = 0)
conditions = [
    (doc['2'] == 0),
    (doc['2'] == 1) | (doc['2'] == 2),
    (doc['2'] == 3)]
choices = ['alone', 'collab', 'diverse']
y = np.select(conditions, choices, default='unknown')
n_channels = X[0].shape[0] # shape of first timeseries is shape of all timeseries

# choose only drawing alone and collaborative drawing
X = [i for idx, i in enumerate(X) if y[idx] != 'diverse']
y = y[y != 'diverse']

print(
    f"Data loaded: {len(X)} trials, {X[0].shape[0]} channels"
)

# ------------------------------------------------------------
### Set up the pipeline

# Define the pipeline with HybridBlocks and Riemannian Lloyd's algorithm
pipeline_Riemannian = Pipeline(
    [
        ("windows", ListTimeSeriesWindowTransformer(window_size=upsampling_freq*15)),
        ("block_kernels", HybridBlocks(block_size=block_size, shrinkage=0.1, metrics="cov")),
        ("kmeans", SVC(metric="riemann", C=1))
    ], verbose=True
)

# Wrap the pipeline so labels are transformed automatically
pipeline_Riemannian = TransformedTargetClassifier2(
    classifier=pipeline_Riemannian,
    transformer=FunctionTransformer(lambda y: np.concatenate([WindowTransformer.transform_labels(y_i) for y_i in y]))
)
# ------------------------------------------------------------
### Fit the models
# Define cross-validation
cv = StratifiedKFold(
    n_splits=cv_splits,
    random_state=random_state,
    shuffle=True
    )

cv_scores_hybrid_blocks = cross_val_score(
    pipeline_Riemannian, X, y,
    cv=cv, scoring="accuracy", n_jobs=n_jobs
    )

### Set up the pipeline

# ------------------------------------------------------------
### Grid search

# Define grid search
#param_grid_hybrid_blocks = {
#    'block_kernels__shrinkage': [0, 0.01, 0.1],
#    'block_kernels__metrics': ['cov', 'rbf', 'lwf', 'tyl', 'corr'],
#    'classifier__C': [0.1, 1, 2],       # Regularization parameter
#    'classifier__metric': ['riemann']
#}

#param_grid_blockcovariances = {
#    'classifier__C': [0.1, 1, 10],       # Regularization parameter
#    'shrinkage__shrinkage': [0.01, 0.1]
#}
#grid_search_hybrid_blocks = GridSearchCV(pipeline_hybrid_blocks, 
#                           param_grid_hybrid_blocks, 
#                           cv=cv, 
#                           scoring='accuracy', 
#                           n_jobs=n_jobs,
#                           verbose=10)

# execute grid search
#print("executing grid search")
#grid_search_hybrid_blocks.fit(X, y)
##grid_search_blockcovariances.fit(X, y)
#print("collecting garbage")
#gc.collect()

# save results
#print("saving results")
#timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#results_hybrid_blocks = pd.DataFrame(grid_search_hybrid_blocks.cv_results_)
#results_hybrid_blocks.to_csv(f"results/grid_search_hybrid_blocks_results_{timestamp}.csv", index=False)
#results_blockcovariances = pd.DataFrame(grid_search_blockcovariances.cv_results_)
#results_blockcovariances.to_csv("grid_search_blockcovariances_results.csv", index=False)
