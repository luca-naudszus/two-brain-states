# ------------------------------------------------------------
# Import packages


from joblib import Parallel, delayed
#---
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
#---
from pyriemann.clustering import Kmeans
from pyriemann.estimation import (
    Covariances,
    Kernels,
    Shrinkage,
)
from pyriemann.utils.base import expm, invsqrtm, logm, sqrtm
from pyriemann.utils.covariance import cov_est_functions
from pyriemann.utils.distance import distance_riemann, distance_wasserstein, pairwise_distance
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

# ------------------------------------------------------------
# define constant values
n_jobs = -1 # use all available cores
max_iter = 5 # maximum number of iterations
random_state = 42
ker_est_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]

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

    def transform(self, X, is_labels, n_windows):
        if not is_labels:
            assert isinstance(X, np.ndarray), "Input must be a NumPy array"
            assert X.ndim == 2, "Unexpected input shape" 
            if self.step_size is None: 
                truncated_length = n_windows * self.window_size
                n_channels = X.shape[0]
                X_windows = X[:, :truncated_length].reshape(n_channels, n_windows, self.window_size)
                X_windows = np.transpose(X_windows, (1, 0, 2))
            else: 
                n_samples = X.shape[1]
                X_windows = np.array([X[:,i:i + self.window_size] for i in range(0, n_samples - self.window_size + 1, self.step_size)])
        else:
            X_windows = n_windows * [X]
        return X_windows

class ListTimeSeriesWindowTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list of time series into non-overlapping windows of shape (n_channels, window_size). 
        Author: Luca Naudszus."""
    def __init__(self, window_size = 75, step_size=None):
        self.window_size = window_size
        self.step_size = step_size
        self.base_transformer = WindowTransformer(window_size, step_size)

    def fit(self, X, y=None):
        assert isinstance(X, list), "Input must be a list of NumPy arrays"
        list_windows = []
        for x in X: 
            n_samples = x.shape[1]
            if self.step_size is None: 
                n_windows = n_samples // self.window_size
            else: 
                n_windows = (n_samples - self.window_size) // self.step_size + 1
            list_windows.append(n_windows)
        self.list_windows_ = list_windows
        return self
    
    def transform(self, X, is_labels=False):
        if not hasattr(self, 'list_windows_'):
            raise ValueError("The transformer must be fitted before calling transform.")
        transformed_x = [self.base_transformer.transform(x, is_labels, self.list_windows_[i]) for i, x in enumerate(X)]
        if not transformed_x:
            raise ValueError("No valid windows found. Check your input time series.")
        return np.concatenate(transformed_x, axis = 0)
    
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
    Adaptations by Luca A. Naudszus. 
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
        print(f'Estimating block kernels with kernel(s) {self.metrics} and shrinkage {self.shrinkage}')

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

class Demeaner(BaseEstimator, TransformerMixin):
    """Demeans SPD matrices. 
        Author: Luca Naudszus."""
    def __init__(self, groups, method="tangent", activate=False):
        self.activate = activate
        self.groups = np.array(groups)
        self.method = method # "log-euclidean", "tangent", "projection", "airm"

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.activate:
            methods = {
                "log-euclidean": log_euclidean_centering,
                "tangent": tangent_space_centering,
                "projection": euclidean_centering_with_projection,
                "airm": airm_centering,
            }

            if self.method not in methods:
                raise ValueError("Invalid method. Choose from 'log-euclidean', 'tangent', 'projection', or 'airm'.")

            print(f"Demeaning matrices using {self.method} centering")
            unique_labels = np.unique(self.groups)
            demeaned_matrices = np.empty_like(X)
            
            for label in unique_labels:
                indices = np.where(self.groups == label)[0]
                group_matrices = X[indices]
                if self.method == 'log-euclidean':
                    demeaned_group = log_euclidean_centering(group_matrices)
                elif self.method == 'tangent':
                    demeaned_group = tangent_space_centering(group_matrices)
                elif self.method == 'projection':
                    demeaned_group = euclidean_centering_with_projection(group_matrices)
                elif self.method == 'airm':
                    demeaned_group = airm_centering(group_matrices)
                else:
                    raise ValueError("Invalid method. Choose from 'log-euclidean', 'tangent', 'projection', or 'airm'.")
                demeaned_matrices[indices] = demeaned_group
            
            self.matrices_ = demeaned_matrices
            
            return demeaned_matrices
        else:
            return X

class RiemannianKMeans(BaseEstimator, ClusterMixin):
    """Wrapper for pyriemann.clustering.KMeans.  
        Author: Luca Naudszus."""
    def __init__(self, n_clusters = 3, n_jobs = n_jobs, max_iter = max_iter, n_init = 100):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.n_init = n_init
        self.metric = "riemann"
        self.kmeans = Kmeans(n_clusters=n_clusters, 
            n_jobs = n_jobs, 
            max_iter = max_iter,
            n_init = n_init,
            metric = "riemann")

    def centroids(self):
        return self.kmeans.centroids()     

    def fit(self, X, y=None):
        print(f"KMeans: Fitting data with shape {X.shape}, using {self.n_init} random initializations")
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_
        return self

    def predict(self, X):
        print(f"KMeans: Predicting data with shape {X.shape}")
        return self.kmeans.predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

def riemannian_silhouette_score(matrices, labels, n_jobs = n_jobs, distance=distance_riemann): 
    """Calculates Silhouette Coefficient based on pairwise Affine-Invariant Riemannian Metric 
    on 10% random matrices (stratified sampling).""" 
    # We could also use Bures-Wasserstein distance (distance_wasserstein), which is a lot faster, 
    # but not what is used in the actual Lloyd's algorithm as implemented above. 
    
    df = pd.DataFrame({'index': np.arange(len(matrices)), 'cluster': labels})
    # sample 10% from each cluster
    stratified_subset = (   
        df.groupby('cluster', group_keys=False)
        .apply(lambda x: x.sample(frac=.1, random_state=random_state), include_groups=False)
        .reset_index(drop=True)
    )
    stratified_indices = stratified_subset['index'].values  # Remove cluster column
    stratified_matrices = matrices[stratified_indices]
    stratified_labels = labels[stratified_indices]
    n_matrices = len(stratified_matrices)

    print(f"SilhouetteScore: Processing stratified sample of {n_matrices} data points")
    
    # (3) Parallel execution of distance calculation
    pairwise_distances_matrix = pairwise_distance(stratified_matrices, metric=distance)

    # (5) Calculate silhouette score
    score = silhouette_score(pairwise_distances_matrix, stratified_labels, metric='precomputed')
    return score

def ch_score(matrices, labels):
    # (1) Extract and transform matrices
    flattener = FlattenTransformer()
    flattener.fit(matrices)
    matrices = flattener.transform(matrices)
    
    # (2) Calculate Calinski-Harabasz score
    score = calinski_harabasz_score(matrices, labels)
    return score

def riemannian_variance(cluster, mean):
    """Computes Riemannian variance as the average squared geodesic distance to the cluster mean."""
    distances = np.array([distance_riemann(X, mean) ** 2 for X in cluster])
    return np.mean(distances)

def geodesic_distance_ratio(clusters, means):
    """Computes the ratio of mean within-cluster to between-cluster geodesic distances."""
    within_distances = []
    between_distances = []
    
    for i, cluster in enumerate(clusters):
        within_distances.extend([distance_riemann(X, means[i]) for X in cluster])
        
        for j in range(len(means)):
            if i != j:
                between_distances.append(distance_riemann(means[i], means[j]))
    
    return np.mean(within_distances) / np.mean(between_distances)

def riemannian_davies_bouldin(clusters, means):
    """Computes the Davies-Bouldin Index using Riemannian variance and geodesic distances."""
    k = len(means)
    sigma = [riemannian_variance(cluster, means[i]) for i, cluster in enumerate(clusters)]
    db_values = []
    
    for i in range(k):
        max_ratio = max(
            (sigma[i] + sigma[j]) / distance_riemann(means[i], means[j])
            for j in range(k) if j != i
        )
        db_values.append(max_ratio)
    
    return np.mean(db_values)

def tangent_space_centering(spd_matrices, reference=None):
    """
    Demeans SPD matrices in the tangent space at a chosen reference matrix.

    Parameters:
    - spd_matrices: list of (n, n) SPD matrices.
    - reference: (n, n) SPD matrix (optional). Defaults to Euclidean mean.

    Returns:
    - Centered SPD matrices.
    """
    n_matrices = len(spd_matrices)
    # Compute reference if not provided (Euclidean mean by default)
    if reference is None:
        reference = sum(spd_matrices) / n_matrices
    ref_sqrt = sqrtm(reference)
    ref_inv_sqrt = np.linalg.inv(ref_sqrt)
    # Project matrices to tangent space
    log_matrices = [ref_inv_sqrt @ logm(ref_inv_sqrt @ X @ ref_inv_sqrt) @ ref_inv_sqrt for X in spd_matrices]
    # Compute mean in tangent space
    log_mean = sum(log_matrices) / n_matrices
    # Demean and project back to SPD space
    spd_centered = [ref_sqrt @ expm(ref_sqrt @ (log_X - log_mean) @ ref_sqrt) @ ref_sqrt for log_X in log_matrices]
    return spd_centered

def project_to_spd(matrix):
    """
    Projects a matrix onto the SPD space by setting negative eigenvalues to a small positive value.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-6)  # Ensure SPD by making all eigenvalues positive
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def euclidean_centering_with_projection(spd_matrices):
    """
    Demeans SPD matrices using Euclidean mean and projects back to SPD space.

    Parameters:
    - spd_matrices: list of (n, n) SPD matrices.

    Returns:
    - Centered SPD matrices.
    """
    n_matrices = len(spd_matrices)
    # Compute Euclidean mean
    mean_matrix = sum(spd_matrices) / n_matrices
    # Demean and project back to SPD space
    spd_centered = [project_to_spd(X - mean_matrix) for X in spd_matrices]
    return spd_centered

def airm_centering(spd_matrices):
    """
    Demeans SPD matrices using Affine-Invariant Riemannian Mean (AIRM) as the reference.

    Parameters:
    - spd_matrices: list of (n, n) SPD matrices.

    Returns:
    - Centered SPD matrices.
    """
    n_matrices = len(spd_matrices)
    # Compute the Riemannian mean using pyriemann
    reference = mean_riemann(spd_matrices)
    ref_sqrt = sqrtm(reference)
    ref_inv_sqrt = np.linalg.inv(ref_sqrt)
    # Project matrices to tangent space
    log_matrices = [ref_inv_sqrt @ logm(ref_inv_sqrt @ X @ ref_inv_sqrt) @ ref_inv_sqrt for X in spd_matrices]
    # Compute mean in tangent space
    log_mean = sum(log_matrices) / n_matrices
    # Demean and project back to SPD space
    spd_centered = [ref_sqrt @ expm(ref_sqrt @ (log_X - log_mean) @ ref_sqrt) @ ref_sqrt for log_X in log_matrices]
    return spd_centered

def log_euclidean_centering(spd_matrices):
    reference = mean_riemann(spd_matrices)
    ref_log = logm(reference)
    log_matrices = [logm(X) for X in spd_matrices]
    demeaned_matrices = [log_X - ref_log for log_X in log_matrices]
    spd_centered = [expm(ref_log + matrix) for matrix in demeaned_matrices]
    return spd_centered

def project_to_common_space(matrices, target_dim):
    """
    Projects SPD matrices to a lower-dimensional common space using principal subspace projection.
    
    Parameters:
        cov_matrices (list of np.array): List of SPD matrices of shape (n_matrices, d, d).
        target_dim (int): Target dimension for projection.
    
    Returns:
        np.array: Projected SPD matrices of shape (n_matrices, target_dim, target_dim).
    """
    d = matrices[0].shape[0]  # original dimension
    n_matrices = len(matrices)
    mean_cov = sum(matrices) / n_matrices

    # transform into Euclidean space
    mean_sqrt_inv = invsqrtm(mean_cov) # whitening transformation
    log_matrices = [logm(mean_sqrt_inv @ C @ mean_sqrt_inv) for C in matrices] #log-map transformation
    
    # flatten SPD matrices for PCA
    flat_matrices = [log_C[np.triu_indices(d)] for log_C in log_matrices]  # Use upper triangular values
    flat_matrices = np.array(flat_matrices)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=(target_dim * (target_dim + 1)) // 2)
    reduced_matrices = pca.fit_transform(flat_matrices)
    
    # reconstruct SPD matrices in the lower-dimensional space
    projected_matrices = []
    for i in range(n_matrices):
        log_proj = np.zeros((target_dim, target_dim))
        log_proj[np.triu_indices(target_dim)] = reduced_matrices[i]
        log_proj = log_proj + log_proj.T - np.diag(np.diag(log_proj))  # ensure symmetry
        projected_matrices.append(expm(log_proj))  # map back to SPD space
    
    return np.array(projected_matrices)