# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 18 February 2025

import pandas as pd
import re
from itertools import compress, chain
from collections import defaultdict
from copy import deepcopy
import numpy as np
from pprint import pprint
import os
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Import MNE processing
from mne.viz import plot_compare_evokeds
from mne import Epochs, events_from_annotations, set_log_level

# Import MNE-NIRS processing
# from mne_nirs.channels import get_long_channels
# from mne_nirs.channels import picks_pair_to_idx
# from mne_nirs.datasets import fnirs_motor_group
from mne.preprocessing.nirs import beer_lambert_law, optical_density,\
    temporal_derivative_distribution_repair, scalp_coupling_index
from mne_nirs.signal_enhancement import enhance_negative_correlation,\
    short_channel_regression

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
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

block_size = 62 # number of channels for HbO and HbR
n_jobs = -1 # use all available cores
cv_splits = 5 # number of cross-validation folds
random_state = 42 # random state for reproducibility


def individual_analysis(bids_path):

    # Read data with annotations in BIDS format
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    #raw_intensity = get_long_channels(raw_intensity, min_dist=0.01)

    # Convert signal to optical density and determine bad channels
    raw_od = optical_density(raw_intensity)
    
    # Apply short-channel regression
    od_corrected = short_channel_regression(raw_od)
    
    # Downsample and apply signal cleaning techniques
    #od_corrected.resample(3)
    od_corrected = temporal_derivative_distribution_repair(od_corrected)
    #sci = scalp_coupling_index(raw_od, h_freq=1.35, h_trans_bandwidth=0.1)
    #raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    #raw_od.interpolate_bads()


    # Convert to haemoglobin and filter
    raw_haemo = beer_lambert_law(od_corrected, ppf=0.1)
    raw_haemo = raw_haemo.filter(0.02, 0.4,
                                 h_trans_bandwidth=0.1,
                                 l_trans_bandwidth=0.01,
                                 verbose=False)

    # Apply further data cleaning techniques and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)
    
    # Extract events 
    events, event_dict = events_from_annotations(raw_haemo, verbose=False,
                                                 #remove "start" trigger (5)
                                                 regexp='^(?![5]).*$')
    
     # set group variable
    group = []
    #if ID.startswith('2'): group ="synchronised"
    if re.match(r'^2', ID): group ="synchronised"
    else: group = "control"
    
    epochs = Epochs(raw_haemo, events,
                    event_id=event_dict,
                    tmin=-5, tmax=30,
                    reject=dict(hbo=600e-6),
                    reject_by_annotation=True,
                    proj=True, baseline=(None, 0),
                    detrend=1,
                    preload=True, verbose=False)

    return raw_haemo, epochs




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
# Load data
participants= chain(range(101,131), range(201, 231))

X = []
annotations = []
for sub in participants: 
    ID = '%02d' % sub 

    # Create path to file based on experiment info
    bids_path = BIDSPath(subject=ID,
                         task="pilotAudFX",
                         root='data/fNIRS_1stLevel',
                         datatype="nirs",
                         suffix="nirs",
                         extension=".snirf")

    # Analyse data and return both ROI and channel results
    data, epochs = individual_analysis(bids_path)

    # Concatenate all participants
    data = epochs.get_data()
    data_reordered = np.concatenate((data[:, ::2, :], data[:, 1::2, :]), axis=1)
    X.append(data_reordered)
    annotations.append(epochs.get_annotations_per_epoch())
X = np.concatenate(X, axis=0)
y = np.concatenate([[entry[2] for inner_list in sublist for entry in inner_list] for sublist in annotations], axis=0)

print(
    f"Data loaded: {X.shape[0]} trials, {X.shape[1]} channels, "
    f"{X.shape[2]} time points"
)

# ------------------------------------------------------------
### Set up the pipeline

# Define the pipeline with HybridBlocks and SVC classifier
pipeline_hybrid_blocks = Pipeline(
    [
        ("block_kernels", HybridBlocks(block_size=block_size)),
        ("classifier", SVC(metric="riemann", C=0.1)),
    ]
)

# Define the pipeline with BlockCovariances and SVC classifier
pipeline_blockcovariances = Pipeline(
    [
        ("covariances", BlockCovariances(block_size=block_size)),
        ('shrinkage', Shrinkage()),
        ("classifier", SVC(metric="riemann", C=0.1)),
    ]
)

# Define the hyperparameters for fitting
pipeline_hybrid_blocks.set_params(
    block_kernels__shrinkage=[0.01, 0],
    block_kernels__metrics=['cov', 'rbf']
    )

pipeline_blockcovariances.set_params(
    covariances__estimator='lwf',
    shrinkage__shrinkage=0.01
    )

# Define cross-validation
cv = StratifiedKFold(
    n_splits=cv_splits,
    random_state=random_state,
    shuffle=True
    )

### Print and plot results
acc_hybrid_blocks = np.mean(cv_scores_hybrid_blocks)
acc_blockcovariances = np.mean(cv_scores_blockcovariances)

print(f"Mean accuracy for HybridBlocks: {acc_hybrid_blocks:.2f}")
print(f"Mean accuracy for BlockCovariances: {acc_blockcovariances:.2f}")

# plot a scatter plot of CV and median scores
plt.figure(figsize=(6, 6))
plt.scatter(cv_scores_hybrid_blocks, cv_scores_blockcovariances)
plt.plot([0.4, 1], [0.4, 1], "--", color="black")
plt.xlabel("Accuracy HybridBlocks")
plt.ylabel("Accuracy BlockCovariances")
plt.title("Comparison of HybridBlocks and Covariances")
plt.legend(["CV Fold Scores"])
plt.show()

# ------------------------------------------------------------
### Grid search

# Define grid search
param_grid_hybrid_blocks = {
    'block_kernels__shrinkage': [0, 0.01, 0.02, 0.1, 0.3, 0.7],
    'block_kernels__metrics': ['cov', 'rbf', 'lwf', 'tyl', 'corr'],
    'classifier__C': [0.1, 1, 10],       # Regularization parameter
    'classifier__metric': ['riemann']
}

#param_grid_blockcovariances = {
#    'classifier__C': [0.1, 1, 10],       # Regularization parameter
#    'shrinkage__shrinkage': [0.01, 0.1]
#}

grid_search_hybrid_blocks = GridSearchCV(pipeline_hybrid_blocks, 
                           param_grid_hybrid_blocks, 
                           cv=cv, 
                           scoring='accuracy', 
                           n_jobs=n_jobs,
                           verbose=10)

# execute grid search
print("executing grid search")
grid_search_hybrid_blocks.fit(X, y)

results = pd.DataFrame(grid_search_hybrid_blocks.cv_results_)