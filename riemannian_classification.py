# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 13 January 2025

import os
from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline
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

block_size = 8 # number of channels for HbO and HbR
n_jobs = -1 # use all available cores
cv_splits = 5 # number of cross-validation folds
random_state = 42 # random state for reproducibility

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

### Load data
# Load the dataset
X = np.load('preprocessed_data.npy')
doc = pd.read_csv('documentation.csv', index_col = 0)
conditions = [
    (doc['2'] == 0),
    (doc['2'] == 1) | (doc['2'] == 2),
    (doc['2'] == 3)]
choices = ['alone', 'collab', 'diverse']
y = np.select(conditions, choices, default='unknown')

print(
    f"Data loaded: {X.shape[0]} trials, {X.shape[1]} channels, "
    f"{X.shape[2]} time points"
)

# Get trials with the label "collab" for "collaboration"
MT_label = "collab"
MT_trials_indices = np.where(y == MT_label)[0]

# Average the data across the "collab" trials
X_MT_erp = np.mean(X[MT_trials_indices, :, :], axis=0)

# select example channel
channel_index = 2

# Plot the averaged signals
plt.figure(figsize=(10, 5))
plt.plot(X_MT_erp[channel_index, :], label="HbO P1", color="red")
plt.plot(X_MT_erp[channel_index + 4, :], label="HbR P1", color="darkred")
plt.plot(X_MT_erp[channel_index + 8, :], label="HbO P2", color="blue")
plt.plot(X_MT_erp[channel_index + 12, :], label="HbR P2", color="darkblue")
plt.xlabel("Time samples")
plt.ylabel("Signal Amplitude")
plt.title(f"ERP for collaboration trials in channel {channel_index}")
plt.legend()
plt.show()

### Set up the pipeline

# Define the pipeline with HybridBlocks and SVC classifier
pipeline_hybrid_blocks = Pipeline(
    [
        ("block_kernels", HybridBlocks(block_size=block_size)),
#        ("classifier", SVC(metric="riemann", C=1)),
        ("classifier", SVC(metric="riemann"))
    ]
)

# Define the pipeline with BlockCovariances and SVC classifier
pipeline_blockcovariances = Pipeline(
    [
        ("covariances", BlockCovariances(block_size=block_size)),
        ('shrinkage', Shrinkage()),
#        ("classifier", SVC(metric="riemann", C=1)),
        ("classifier", SVC(metric="riemann"))
    ]
)

# Define the hyperparameters for fitting
#pipeline_hybrid_blocks.set_params(
#    block_kernels__shrinkage=[0.02, 0.02], 
#    block_kernels__metrics= ['cov', 'rbf']
#    )

pipeline_blockcovariances.set_params(
    covariances__estimator='lwf',
#   shrinkage__shrinkage=0.0001
    )

# Define cross-validation
cv = StratifiedKFold(
    n_splits=cv_splits,
    random_state=random_state,
    shuffle=True
    )

### Fit the two models
#cv_scores_hybrid_blocks = cross_val_score(
#    pipeline_hybrid_blocks, X, y,
#    cv=cv, scoring="accuracy", n_jobs=n_jobs
#    )

#cv_scores_blockcovariances = cross_val_score(
#    pipeline_blockcovariances, X, y,
#    cv=cv, scoring="accuracy", n_jobs=n_jobs)

### Print and plot results
#acc_hybrid_blocks = np.mean(cv_scores_hybrid_blocks)
#acc_blockcovariances = np.mean(cv_scores_blockcovariances)

#print(f"Mean accuracy for HybridBlocks: {acc_hybrid_blocks:.2f}")
#print(f"Mean accuracy for BlockCovariances: {acc_blockcovariances:.2f}")

# plot a scatter plot of CV and median scores
#plt.figure(figsize=(6, 6))
#plt.scatter(cv_scores_hybrid_blocks, cv_scores_blockcovariances)
#plt.plot([0.4, 1], [0.4, 1], "--", color="black")
#plt.xlabel("Accuracy HybridBlocks")
#plt.ylabel("Accuracy BlockCovariances")
#plt.title("Comparison of HybridBlocks and Covariances")
#plt.legend(["CV Fold Scores"])
#plt.show()

# Define grid search
param_grid_hybrid_blocks = {
    'block_kernels__shrinkage': [0.01, 0.1],
    'block_kernels__metrics': ['cov', 'rbf'],
    'classifier__C': [0.1, 1, 10],       # Regularization parameter
}

param_grid_blockcovariances = {
    'classifier__C': [0.1, 1, 10],       # Regularization parameter
    'shrinkage__shrinkage': [0.01, 0.1]
}

grid_search_hybrid_blocks = GridSearchCV(pipeline_hybrid_blocks, 
                           param_grid_hybrid_blocks, 
                           cv=cv, 
                           scoring='accuracy', 
                           n_jobs=n_jobs)

grid_search_blockcovariances = GridSearchCV(pipeline_blockcovariances, 
                           param_grid_blockcovariances, 
                           cv=cv, 
                           scoring='accuracy', 
                           n_jobs=n_jobs)

# execute grid search
grid_search_hybrid_blocks.fit(X, y)
grid_search_blockcovariances.fit(X, y)

# save results
results_hybrid_blocks = pd.DataFrame(grid_search_hybrid_blocks.cv_results_)
results_hybrid_blocks.to_csv("grid_search_hybrid_blocks_results.csv", index=False)
results_blockcovariances = pd.DataFrame(grid_search_blockcovariances.cv_results_)
results_blockcovariances.to_csv("grid_search_blockcovariances_results.csv", index=False)
