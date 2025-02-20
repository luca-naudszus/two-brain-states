# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 20 February 2025


# ------------------------------------------------------------
# import packages and custom functions

from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from riemannianKMeans import ListTimeSeriesWindowTransformer, HybridBlocks, RiemannianKMeans, ch_score, riemannian_silhouette_score

# ------------------------------------------------------------
# define single analysis
def pipeline(X, y, sessions, dyad, plot, window_length, step_length, shrinkage, metrics, n_clusters): 
    # choose only one dyad
    indices = np.where(dyads == dyad)[0]
    X_dyad, y_dyad, sessions_dyad = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices]
    print(
        f"Data loaded for dyad {dyad}: {len(X_dyad)} trials, {X_dyad[0].shape[0]} channels"
    )
    ### Set up the pipeline

    # Define the pipeline with HybridBlocks and Riemannian Lloyd's algorithm
    pipeline_Riemannian = Pipeline(
        [
            ("windows", ListTimeSeriesWindowTransformer(
                window_size = upsampling_freq*window_length,
                step_size = upsampling_freq*step_length
            )),
            ("block_kernels", HybridBlocks(block_size=block_size,
                                       shrinkage=shrinkage, 
                                       metrics=metrics
            )),
            ("kmeans", RiemannianKMeans(n_jobs=n_jobs,
                n_clusters = n_clusters, 
                n_init = n_init))
        ], verbose = True
    )

    # ------------------------------------------------------------
    ### Fit the models
    pipeline_Riemannian.fit(X_dyad)
    sh_score_pipeline = riemannian_silhouette_score(pipeline_Riemannian)
    ch_score_pipeline = ch_score(pipeline_Riemannian)
    print(f"Silhouette Score: {sh_score_pipeline}")
    print(f"Calinski-Harabasz Score: {ch_score_pipeline}")

    # # ------------------------------------------------------------
    ### Predict labels and calculate Rand score
    matrices = np.array(pipeline_Riemannian.named_steps["block_kernels"].matrices_)
    classes = pipeline_Riemannian.named_steps["kmeans"].predict(matrices)
    activities = pipeline_Riemannian.named_steps["windows"].transform_labels(X_dyad, y_dyad)
    sessions = pipeline_Riemannian.named_steps["windows"].transform_labels(X_dyad, sessions_dyad)
    rand_score_act = rand_score(classes, activities)
    rand_score_ses = rand_score(classes, sessions)

    # ------------------------------------------------------------
    ### PCA 
    mean_matrix = mean_riemann(matrices)
    X_tangent = tangent_space(matrices, mean_matrix)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tangent)

    # Plot
    if plot == 1: 
        # With classes as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(classes): 
            plt.scatter(X_pca[classes == label, 0], X_pca[classes == label, 1], label=f"Class {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for dyad {dyad}")
        plt.legend()
        plt.show()
        # With activity as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(activities): 
            plt.scatter(X_pca[activities == label, 0], X_pca[activities == label, 1], label=f"Class {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for dyad {dyad}")
        plt.legend()
        plt.show()
        # With session as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(sessions): 
            plt.scatter(X_pca[sessions == label, 0], X_pca[sessions == label, 1], label=f"Class {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for dyad {dyad}")
        plt.legend()
        plt.show()
    
    return matrices, classes, sh_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, len(sessions_dyad)

# ------------------------------------------------------------
### set arguments

# which dyad do we want to look at? (set dyad = 0 for all dyads)
dyad = 2014

# do we want to do a single run or a grid search? (0 = single run, 1 = grid search)
grid_search = 0
## in case of 0, define hyperparameters below
## in case of 1, define parameter space below

# are we interested in the plot? (0/1, overridden in case of grid search)
plot = 1

# define global settings
n_jobs = -1 # use all available cores
random_state = 42 # random state for reproducibility
n_init = 10 # number of initializations for kMeans
max_iter = 5 # maximum number of iterations for kMeans

# hyperparameters (overridden in case of grid search)
cv_splits = 5 # number of cross-validation folds
shrinkage = 0.01 # shrinkage value
metrics = 'cov' # kernel function
n_clusters = 3 # number of clusters for k-means

# parameter space for grid search
params_shrinkage = [0, 0.01, 0.1]
params_kernel = ['cov', 'rbf', 'lwf', 'tyl', 'corr']
params_n_clusters = range(3, 8)

# information on data
block_size = 8 # number of channels for HbO and HbR
upsampling_freq = 5 # frequency to which the data have been upsampled
window_length = 15 # length of windows in s
step_length = 1 # steps 

# ------------------------------------------------------------
### Load data

# Load the dataset
npz = np.load('./data/ts_four_blocks.npz')
X = []
for array in list(npz.files):
    X.append(npz[array])
doc = pd.read_csv('./data/doc_four_blocks.csv', index_col = 0)
conditions = [
    (doc['2'] == 0),
    (doc['2'] == 1) | (doc['2'] == 2),
    (doc['2'] == 3)]
choices = ['alone', 'collab', 'diverse']
dyads = np.array(doc['0'])
sessions = np.array(doc['1'])
y = np.select(conditions, choices, default='unknown')
n_channels = X[0].shape[0] # shape of first timeseries is shape of all timeseries

# choose only drawing alone and collaborative drawing
X = [i for idx, i in enumerate(X) if y[idx] != 'diverse']
dyads = dyads[y != 'diverse']
sessions = sessions[y != 'diverse']
y = y[y != 'diverse']

# make variable for chosen dyads
chosen_dyads = np.unique(dyads) if dyad == 0 else [dyad]

# ------------------------------------------------------------
### Run pipeline 


if grid_search == 0:
    scores = []
    for dyad in chosen_dyads: 
        matrices, classes, sh_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, n_sessions = pipeline(X, y, sessions,
            dyad=dyad, plot=plot, window_length=window_length, step_length=step_length, 
            shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
        scores.append(
            [dyad, sh_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, n_sessions]
        )
    scores = pd.DataFrame(scores, columns = [
        'Dyad', 'SilhouetteCoefficient', 'CalinskiHarabaszScore',
        'RandScoreActivities', 'RandScoreSessions', 'nSessions']
    )
        

# ------------------------------------------------------------
### Run grid search

if grid_search == 1:
    # Compute grid search parameters from inputs
    comb_shrinkage = product(params_shrinkage, repeat = int(n_channels / block_size))
    params_shrinkage_combinations = [list(x) for x in comb_shrinkage]
    comb_kernel = product(params_kernel, repeat = int(n_channels / block_size))
    params_kernel_combinations = [list(x) for x in comb_kernel]
    params_n_clusters = range(3,10)
    plot = 0 # do not plot during grid search

    scores = []
    i = 0
    for dyad in chosen_dyads: 
        for shrinkage in params_shrinkage_combinations: 
            for kernel in params_kernel_combinations: 
                for n_clusters in params_n_clusters: 
                    i += 1
                    print(f"Iteration {i}, parameters: dyad {dyad}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                    try:
                        matrices, classes, sh_score_pipeline, ch_score_pipeline, n_sessions = pipeline(X, y, sessions,
                            dyad=dyad, plot=plot, window_length=window_length, step_length=step_length, 
                            shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                    except ValueError as e:
                        print(f"Skipping due to error: {e}")  # Optional: print the error message
                        continue
                    scores.append(
                        [dyad, window_length, shrinkage, kernel, n_clusters, 
                            sh_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, n_sessions])
    scores = pd.DataFrame(scores, columns=['Dyad', 'WindowLength', 'Shrinkage', 'Kernel', 'nClusters', 
                                       'SilhouetteCoefficient', 'CalinskiHarabaszScore',
                                       'RandScoreActivities', 'RandScoreSessions', 'nSessions'])