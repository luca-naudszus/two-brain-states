import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import silhouette_score, calinski_harabasz_score

import pyriemann
from pyriemann.classification import SVC
from pyriemann.estimation import (
    BlockCovariances, 
    Covariances,
    Kernels,
    Shrinkage,
)
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
ker_est_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]
from pyriemann.utils.covariance import cov_est_functions

from scipy.linalg import block_diag

# ------------------------------------------------------------
# define constant values
n_jobs = -1 # use all available cores
max_iter = 5 # maximum number of iterations
random_state = 42

# ------------------------------------------------------------
# define custom classes and functions
class WindowTransformer(BaseEstimator, TransformerMixin):
    """Splits the time series into non-overlapping windows of shape (n_channels, window_size). 
        Author: Luca Naudszus."""
    def __init__(self, window_size=75, step_size=None):
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"
        n_channels = X.shape[0]
        assert X.ndim == 2 and (n_channels in {8, 16}), "Unexpected input shape" #TODO: accept different n_channels
        n_samples = X.shape[1]
        if self.step_size == None: 
            n_windows = n_samples // self.window_size
            truncated_length = n_windows * self.window_size
            X_windows = X[:, :truncated_length].reshape(n_channels, n_windows, self.window_size)
            return np.transpose(X_windows, (1, 0, 2))
        else: 
            n_windows = (n_samples - self.window_size) // self.step_size + 1
            # Extract windows using a sliding approach
            X_windows = np.array([X[:,i:i + self.window_size] for i in range(0, n_samples - self.window_size + 1, self.step_size)])
            return X_windows
        
    def transform_labels(self, X, y): 
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"
        n_channels = X.shape[0]
        assert X.ndim == 2 and (n_channels in {8, 16}), "Unexpected input shape" #TODO: accept different n_channels
        n_samples = X.shape[1]
        if self.step_size == None: 
            n_windows = n_samples // self.window_size
        else: 
            n_windows = (n_samples - self.window_size) // self.step_size + 1
        y = n_windows * [y]
        return y

class ListTimeSeriesWindowTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list of time series into non-overlapping windows of shape (n_channels, window_size). 
        Author: Luca Naudszus."""
    def __init__(self, window_size = 75, step_size=None):
        self.window_size = window_size
        self.step_size = step_size
        self.base_transformer = WindowTransformer(window_size, step_size)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, list), "Input must be a list of NumPy arrays"
        transformed = [self.base_transformer.transform(x) for x in X]
        return np.concatenate(transformed, axis = 0)
    
    def transform_labels(self, X, y):
        """Adjust labels to match the windowed data."""
        transformed_y = [self.base_transformer.transform_labels(x, y_i) for x, y_i in zip(X, y)]
        return np.concatenate(transformed_y, axis = 0)

class Stacker(TransformerMixin):
    """Stacks values of a DataFrame column into a 3D array. Author: Tim Näher."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "Input must be a DataFrame"
        assert X.shape[1] == 1, "DataFrame must have only one column"
        return np.stack(X.iloc[:, 0].values)


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flattens the last two dimensions of an array.
        ColumnTransformer requires 2D output, so this transformer
        is needed. Author: Tim Näher."""
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
    
    Author: Tim Näher.
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
        self.matrices_ = M_matrices
        return M_matrices

    def _prepare_dataframe(self, X):
        """Converts the data into a df with eac hblock as column."""
        data_dict = {}
        for i, (start, end) in enumerate(self.block_indices):
            data_dict[self.block_names[i]] = list(X[:, start:end, :])
        return pd.DataFrame(data_dict)

class RiemannianKMeans(BaseEstimator, ClusterMixin):
    """Wrapper for pyriemann.clustering.KMeans for usage in GridSearchCV. 
        Author: Luca Naudszus."""
    def __init__(self, n_clusters = 3, n_jobs = n_jobs, max_iter = max_iter, n_init = 100):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.n_init = n_init
        self.metric = "riemann"
        self.kmeans = pyriemann.clustering.Kmeans(n_clusters=n_clusters, 
            n_jobs = n_jobs, 
            max_iter = max_iter,
            n_init = n_init,
            metric = "riemann")
        

    def fit(self, X, y=None):
        print(f"Fitting data with shape {X.shape}, using {self.n_init} random initializations")
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_
        return self

    def predict(self, X):
        print(f"Predicting data with shape {X.shape}")
        return self.kmeans.predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

def riemannian_silhouette_score(pipeline, n_jobs = n_jobs, distance=distance_riemann): 
    """Calculates Silhouette Coefficient based on pairwise Affine-Invariant Riemannian Metric 
    on 10% random matrices (stratified sampling).""" 
    # We could also use Bures-Wasserstein distance (distance_wasserstein), which is a lot faster, 
    # but not what is used in the actual Lloyd's algorithm as implemented above. 

    # (1) Extract matrices
    block_matrices = pipeline.named_steps["block_kernels"].matrices_
    labels = pipeline.named_steps["kmeans"].labels_

    # (2) Stratified sampling
    #random_idx = np.random.choice(range(block_matrices.shape[0]), block_matrices.shape[0]/10, replace=False)
    #sampled_matrices = block_matrices[random_idx]
    #sampled_labels = labels[random_idx]
    #n_matrices = sampled_matrices.shape[0]
    
    df = pd.DataFrame({'index': np.arange(len(block_matrices)), 'cluster': labels})
    # sample 10% from each cluster
    stratified_subset = (   
        df.groupby('cluster', group_keys=False)
        .apply(lambda x: x.sample(frac=.1, random_state=random_state), include_groups=False)
        .reset_index(drop=True)
    )
    stratified_indices = stratified_subset['index'].values  # Remove cluster column
    stratified_matrices = block_matrices[stratified_indices]
    stratified_labels = labels[stratified_indices]
    n_matrices = len(stratified_matrices)

    print(f"Processing: stratified sample of {n_matrices} data points")
    
    # (3) Parallel execution of distance calculation
    def compute_distance(i, j, matrices, distance):
        if j == 0:
            print(f"Processing i = {i} (outer loop)")
        return distance(matrices[i], matrices[j])

    pairwise_distances = Parallel(n_jobs=n_jobs)(
        delayed(compute_distance)(i, j, stratified_matrices, distance)
        for i in range(n_matrices) for j in range(i+1, n_matrices)
    )
    
    # (4) Convert list into full distance matrix
    pairwise_distances_matrix = np.zeros((n_matrices, n_matrices))
    idx = np.triu_indices(n_matrices, 1)
    pairwise_distances_matrix[idx] = pairwise_distances
    pairwise_distances_matrix += pairwise_distances_matrix.T

    # (5) Calculate silhouette score
    score = silhouette_score(pairwise_distances_matrix, stratified_labels, metric='precomputed')
    return score

def ch_score(pipeline):
    # (1) Extract and transform matrices
    flattener = FlattenTransformer()
    block_matrices = pipeline.named_steps["block_kernels"].matrices_
    flattener.fit(block_matrices)
    block_matrices = flattener.transform(block_matrices)
    preds = pipeline.named_steps["kmeans"].labels_
    
    # (2) Calculate Calinski-Harabasz score
    score = calinski_harabasz_score(block_matrices, preds)
    return score