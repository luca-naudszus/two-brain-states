# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 20 February 2025


# ------------------------------------------------------------
# import packages and custom functions

from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from riemannianKMeans import Demeaner, FlattenTransformer, ListTimeSeriesWindowTransformer, HybridBlocks, RiemannianKMeans, ch_score, geodesic_distance_ratio, riemannian_davies_bouldin, riemannian_silhouette_score, riemannian_variance

os.chdir('/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code')
# ------------------------------------------------------------
# define single analysis
def pipeline(X, y, id, session, demean, demeaner_var, demeaner_method, plot, window_length, step_length, shrinkage, metrics, n_clusters): 
    
    # ------------------------------------------------------------
    ### Get data
    #TODO: implement channel variable in preparations tool
    if clustering == 'full':
        X_tmp, y_tmp, sessions_tmp, ids_tmp, channels_tmp = X, y, sessions, ids, channels
        print(f"Data loaded: {len(X_tmp)} trials, {X_tmp[0].shape[0]} channels")
    elif clustering == 'id-wise':
        indices = np.where(ids == id)[0]
        X_tmp, y_tmp, sessions_tmp, ids_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [channels[i] for i in indices]
        print(f"Data loaded for id {id}: {len(X_tmp)} trials, {X_tmp[0].shape[0]} channels")
    elif clustering == 'session-wise':
        indices = np.where((ids == id) & (sessions == session))[0]
        X_tmp, y_tmp, sessions_tmp, ids_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [channels[i] for i in indices]
        assert len(indices) != 0, 'ID-session combination does not exist in data set.'
        print(f"Data loaded for id {id} and session {session}: {len(X_tmp)} trials, {X_tmp[0].shape[0]} channels")
    
    # ------------------------------------------------------------
    ### Group by number of channels and segment into windows
    
    grouping_indices = [[np.where(np.array(channels_tmp) == channel_no)[0]] for channel_no in set(channels_tmp)]
    windowsTransformer = ListTimeSeriesWindowTransformer(
                window_size = upsampling_freq*window_length,
                step_size = upsampling_freq*step_length
            )
    
    # Group variables
    X_grouped = [[X_tmp[i] for i in ind[0]] for ind in grouping_indices]
    activities_grouped = [[y_tmp[i] for i in ind[0]] for ind in grouping_indices]
    sessions_grouped = [[sessions_tmp[i] for i in ind[0]] for ind in grouping_indices]
    ids_grouped = [[ids_tmp[i] for i in ind[0]] for ind in grouping_indices]
    
    # Fit transformer
    X_seg, activities_seg, sessions_seg, ids_seg = [], [], [], []
    
    # Transform into windows
    for in_channelgroup in range(0, len(X_grouped)):
        X_seg.append(windowsTransformer.fit_transform(X_grouped[in_channelgroup]))
        activities_seg.append(windowsTransformer.transform(activities_grouped[in_channelgroup]), is_labels=True)
        sessions_seg.append(windowsTransformer.transform(sessions_grouped[in_channelgroup]), islabels=True)
        ids_seg.append(windowsTransformer.transform(ids_grouped[in_channelgroup]), islabels=True)
    
    # Set group variables for demeaner
    if demean: 
        if clustering == 'full': 
            if demeaner_var == 'ids':
               groups = trans_ids
            elif demeaner_var == 'session-wise':
                groups = trans_sessions
            else: 
                print("Warning: will not demean because demeaning mode is unclear.")
        elif clustering == 'id-wise': 
            if demeaner_var == 'session-wise':
                groups = trans_sessions
            elif demeaner_var == 'id-wise': 
                print("Warning: will not demean id-wise because clustering is id-wise.")
            else: 
                print("Warning: will not demean because demeaning mode is unclear.")
        elif clustering == 'session-wise': 
            print("Warning: will not demean because clustering is session-wise.")
    
    # ------------------------------------------------------------
    ### Get kernel matrices with HybridBlocks
    block_kernels = HybridBlocks(block_size=block_size,
                                 shrinkage=shrinkage,
                                 metrics=metrics)
    matrices = [block_kernels.fit_transform(X_seg[in_channelgroup]) for in_channelgroup in range(0, len(X_seg))]

    # ------------------------------------------------------------
    ### TODO: Make array with common dimensions
    X_common = matrices

    # ------------------------------------------------------------
    demeaner = Demeaner(groups=groups,
                        activate=demean,
                        method=demeaner_method)
    matrices = demeaner.fit_transform(X_common)

    # ------------------------------------------------------------
    ### KMeans
    kmeans = RiemannianKMeans(n_jobs=n_jobs,
                              n_clusters=n_clusters,
                              n_init=n_init)
    classes = kmeans.fit_predict(matrices)
    cluster_means = kmeans.centroids()
    clusters = [matrices[classes == i] for i in range(n_clusters)]
    
    # ------------------------------------------------------------
    ### Clustering performance evaluation
    sh_score_pipeline = riemannian_silhouette_score(pipeline_Riemannian)
    ch_score_pipeline = ch_score(pipeline_Riemannian)
    riem_var_pipeline = [riemannian_variance(clusters[i], cluster_means[i]) for i in range(n_clusters)]
    db_score_pipeline = riemannian_davies_bouldin(clusters, cluster_means)
    gdr_pipeline = geodesic_distance_ratio(clusters, cluster_means)
    print(f"Silhouette Score: {sh_score_pipeline}")
    print(f"Calinski-Harabasz Score: {ch_score_pipeline}")
    print(f"Riemannian Variance: {riem_var_pipeline}")
    print(f"Davies-Bouldin-Index: {db_score_pipeline}")
    print(f"Geodesic Distance Ratio: {gdr_pipeline}")

    # # ------------------------------------------------------------
    ### Rand indices
    rand_score_id = adjusted_rand_score(classes, trans_ids) if clustering == 'all' else np.nan
    rand_score_ses = adjusted_rand_score(classes, trans_sessions) if clustering != 'session-wise' else np.nan
    rand_score_act = adjusted_rand_score(classes, trans_activities)
    

    # ------------------------------------------------------------
    ### PCA 
    mean_matrix = mean_riemann(matrices)
    X_tangent = tangent_space(matrices, mean_matrix)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tangent)

    # Plot
    if plot == 1 and grid_search == 0: 
        # With classes as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(classes): 
            plt.scatter(X_pca[classes == label, 0], X_pca[classes == label, 1], label=f"Class {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for ID {id}, clustering")
        plt.legend()
        plt.show()
        
        # With session as labels
        if clustering != 'session-wise':
            plt.figure(figsize=(6, 5))
            for label in np.unique(trans_sessions): 
                plt.scatter(X_pca[trans_sessions == label, 0], X_pca[trans_sessions == label, 1], label=f"Session: {label}", alpha=0.8)
            plt.xlabel("PC1")
            plt.xlabel("PC2")
            plt.title(f"Tangent Space PCA projection for ID {id}, sessions")
            plt.legend()
            plt.show()

        # With activity as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(trans_activities)[::-1]: 
            plt.scatter(X_pca[trans_activities == label, 0], X_pca[trans_activities == label, 1], label=f"Activity: {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for ID {id}, activities")
        plt.legend()
        plt.show()
    
    return matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, len(sessions_tmp)

# ------------------------------------------------------------
### set arguments

# which type of data are we interested in?
type_of_data = "two_blocks"
# one_brain, two_blocks, four_blocks: channel-wise z-scoring
# one_brain_session etc.: channel- and session-wise z-scoring

# how do we want to cluster?
# Choose from 'full', 'id-wise', 'session-wise'
clustering = 'id-wise' 
# Which dyad/participant do we want to look at? (only for id-wise and session-wise clustering)
which_id = 2014 # set which_id = 'all' for all dyads/participants
# Which session do we want to look at? (only for session-wise clustering)
which_session = 5 # set which_session = 'all' for all sessions

# should the matrices be demeaned? 
demean = True
# if so, within-id or within-session?
demeaner_var = 'session-wise' # 'none', 'id-wise', 'session-wise'
# if so, which method?
demeaner_method = 'airm' # 'log-euclidean', 'tangent', 'projection', or 'airm'
# 'projection' takes the longest, but seems to give the best result

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
metrics = 'rbf' # kernel function
n_clusters = 3 # number of clusters for k-means

# parameter space for grid search
params_shrinkage = [0, 0.01, 0.1]
params_kernel = ['cov', 'rbf', 'lwf', 'tyl', 'corr']
params_n_clusters = range(3, 8)

# information on data
block_size = 4 # number of channels for HbO and HbR
upsampling_freq = 5 # frequency to which the data have been upsampled
window_length = 15 # length of windows in s
step_length = 1 # steps 

# ------------------------------------------------------------
### Load data

# Load the dataset
npz = np.load(f"./data/ts_{type_of_data}.npz")
X = []
for array in list(npz.files):
    X.append(npz[array])
doc = pd.read_csv(f"./data/doc_{type_of_data}.csv", index_col = 0)
ids = np.array(doc['0'])
sessions = np.array(doc['1'])
conditions = [
    (doc['2'] == 0),
    (doc['2'] == 1) | (doc['2'] == 2),
    (doc['2'] == 3)]
choices = ['alone', 'collab', 'diverse']
y = np.select(conditions, choices, default='unknown')

# make variable for chosen ids
chosen_ids = np.unique(ids) if which_id == 'all' else [which_id]


# choose only drawing alone and collaborative drawing
X = [i for idx, i in enumerate(X) if y[idx] != 'diverse']
ids = ids[y != 'diverse']
sessions = sessions[y != 'diverse']
y = y[y != 'diverse']

n_channels = X[0].shape[0] # shape of first timeseries is shape of all timeseries

# ------------------------------------------------------------
### Run pipeline 

if grid_search == 0:
    scores = []
    if clustering == 'full':
        matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                X, y, id=np.nan, session=np.nan, 
                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                plot=plot, window_length=window_length, step_length=step_length, 
                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
        scores.append(
                ['all', 'all', sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, 
                 rand_score_act, rand_score_ses, rand_score_id, n_activities]
            )
    elif clustering == 'id-wise':
        for id in chosen_ids: 
            matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                X, y, id=id, session=np.nan, 
                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                plot=plot, window_length=window_length, step_length=step_length, 
                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
            scores.append(
                [id, 'all', sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, 
                 rand_score_act, rand_score_ses, rand_score_id, n_activities]
            )
    elif clustering == 'session-wise':
        for id in chosen_ids:
            # make variable for chosen sessions
            chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]

            for session in chosen_sessions: 
                matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                    X, y, id=id, session=session, 
                    demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                    plot=plot, window_length=window_length, step_length=step_length, 
                    shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                scores.append(
                    [id, session, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, 
                     rand_score_act, rand_score_ses, rand_score_id, n_activities]
                ) 
    scores = pd.DataFrame(scores, columns = [
        'ID', 'Session', 'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio',
        'RandScoreActivities', 'RandScoreSessions', 'RandScoreIds', 'nActivities']
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
    for shrinkage in params_shrinkage_combinations:
        for kernel in params_kernel_combinations:
            for n_clusters in params_n_clusters:
                if clustering == 'full':
                    i += 1
                    print(f"Iteration {i}, parameters: shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                    try:
                        matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, db_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                        X, y, id=np.nan, session=np.nan, 
                        demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                        plot=plot, window_length=window_length, step_length=step_length,
                        shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                    except ValueError as e:
                        print(f"Skipping due to error: {e}")
                        continue
                    scores.append(
                        ['all', 'all', window_length, shrinkage, kernel, n_clusters, 
                         sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline,
                         rand_score_act, rand_score_ses, rand_score_id, n_activities]
                    )
                elif clustering == 'id-wise':
                    for id in chosen_ids:
                        i += 1
                        print(f"Iteration {i}, parameters: ID {id}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                        try:
                            matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, db_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                                X, y, id=id, session=np.nan, 
                                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                plot=plot, window_length=window_length, step_length=step_length,
                                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                        except ValueError as e:
                            print(f"Skipping due to error: {e}")  
                            continue
                        scores.append(
                            [id, 'all', window_length, shrinkage, kernel, n_clusters, 
                             sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline,
                             rand_score_act, rand_score_ses, rand_score_id, n_activities])
                elif clustering == 'session-wise':
                    for id in chosen_ids:
                        chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]
                        for session in chosen_sessions:
                            i += 1
                            print(f"Iteration {i}, parameters: ID {id}, session {session}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                            try:
                                matrices, classes, trans_activities, trans_sessions, sh_score_pipeline, db_score_pipeline, ch_score_pipeline, rand_score_act, rand_score_ses, rand_score_id, n_activities = pipeline(
                                X, y, id=id, session=session, 
                                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                plot=plot, window_length=window_length, step_length=step_length,
                                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                            except ValueError as e:
                                print(f"Skipping due to error: {e}")  # Optional: print the error message
                                continue
                            except AssertionError as e:
                                print(f"Skipping due to error: {e}")  # Optional: print the error message
                                continue
                            scores.append(
                                [id, session, window_length, shrinkage, kernel, n_clusters, 
                                 sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline,
                                rand_score_act, rand_score_ses, rand_score_id, n_activities])
    scores = pd.DataFrame(scores, columns=['ID', 'Session', 'WindowLength', 'Shrinkage', 'Kernel', 'nClusters', 
                                       'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio'
                                       'RandScoreActivities', 'RandScoreSessions', 'RandScoreIds', 'nSessions'])

# ------------------------------------------------------------
### save results
print("saving results")
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
scores.to_csv(f"results/results_{type_of_data}_{timestamp}.csv", index=False)
