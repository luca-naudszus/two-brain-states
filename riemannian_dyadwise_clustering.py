# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 20 February 2025


# ------------------------------------------------------------
# Import packages and custom functions

from datetime import datetime
import json
import os
from pathlib import Path
#---
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#---
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
#---
from riemannianKMeans import (
    Demeaner, 
    ListTimeSeriesWindowTransformer, 
    HybridBlocks, 
    RiemannianKMeans, 
    ch_score, 
    geodesic_distance_ratio, 
    project_to_common_space, 
    riemannian_davies_bouldin, 
    riemannian_silhouette_score, 
    riemannian_variance
)

# ------------------------------------------------------------
# Set path
os.chdir('/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code')
outpath = 'results'

# ------------------------------------------------------------
# Set arguments. Change only variables in this section of the script. 

# which type of data are we interested in?
type_of_data = "one_brain"
# one_brain, two_blocks, four_blocks: channel-wise z-scoring
# one_brain_session etc.: channel- and session-wise z-scoring
exp_block_size = 4 
which_freq_bands = 0 # Choose from 0 (0.01 to 0.4), 1 (0.1 to 0.2), 2 (0.03 to 0.1), 3 (0.02 to 0.03). 

# do we want to use data with missing channels?
use_missing_channels = False
# if so, data from all sessions are projected into a common space

# how do we want to cluster?
# Choose from 'full', 'id-wise', 'session-wise'
clustering = 'id-wise' 
# Which dyad/participant do we want to look at? (only for id-wise and session-wise clustering)
which_id = 105 # set which_id = 'all' for all dyads/participants
# Which session do we want to look at? (only for session-wise clustering)
which_session = 'all' # set which_session = 'all' for all sessions

# should the matrices be demeaned? 
demean = False
# if so, within-id or within-session?
demeaner_var = 'session-wise' # 'none', 'id-wise', 'session-wise'
# if so, which method?
demeaner_method = 'projection' # 'log-euclidean', 'tangent', 'projection', or 'airm'
# 'projection' takes the longest, but seems to give the best result

# do we want to do a single run or a grid search? (False = single run, True = grid search)
grid_search = False
## in case of False, define hyperparameters below
## in case of True, define parameter space below

# are we interested in the plot? (True/False, overridden in case of grid search: no plot)
plot = False

# hyperparameters (overridden in case of grid search)
shrinkage = 0.01 # shrinkage value
metrics = 'cov' # kernel function
n_clusters = 5 # number of clusters for k-means

# parameter space for grid search
params_shrinkage = [0, 0.01, 0.1]
params_kernel = ['cov', 'rbf', 'lwf', 'tyl', 'corr']
params_n_clusters = range(3, 10)

# information on data
upsampling_freq = 5 # frequency to which the data have been upsampled
window_length = 30 # length of windows in s
step_length = 1 # steps 

# define global settings
n_jobs = -1 # use all available cores
random_state = 42 # random state for reproducibility
n_init = 10 # number of initializations for kMeans
max_iter = 5 # maximum number of iterations for kMeans

# ------------------------------------------------------------
# Define custom functions. Do not change this section. 

def pipeline(X, y, id, session, demean, demeaner_var, demeaner_method, plot, window_length, step_length, shrinkage, metrics, n_clusters): 
    
    # ------------------------------------------------------------
    ### Get data
    if clustering == 'full':
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = X, y, sessions, ids, blocks, n_channels
        print(f"Data loaded: {len(X_tmp)} trials")
    elif clustering == 'id-wise':
        indices = np.where(ids == id)[0]
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [blocks[i] for i in indices], [n_channels[i] for i in indices]
        print(f"Data loaded for id {id}: {len(X_tmp)} trials")
    elif clustering == 'session-wise':
        indices = np.where((ids == id) & (sessions == session))[0]
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [blocks[i] for i in indices], [n_channels[i] for i in indices]
        assert len(indices) != 0, 'ID-session combination does not exist in data set.'
        print(f"Data loaded for id {id} and session {session}: {len(X_tmp)} trials, {X_tmp[0].shape[0]} channels")
    
    # ------------------------------------------------------------
    ### Group by number of channels and segment into windows
    print("Preparing data")
    if use_missing_channels: 
        grouping_indices = [[np.where(np.array(channels_tmp) == channel_no)[0]] for channel_no in set(channels_tmp)]
    else:
        grouping_indices = [np.where(np.array(channels_tmp) == exp_n_channels)]
    windowsTransformer = ListTimeSeriesWindowTransformer(
            window_size = upsampling_freq*window_length,
            step_size = upsampling_freq*step_length
        )
    
    # Group variables
    X_grouped = [[X_tmp[i] for i in ind[0]] for ind in grouping_indices]
    activities_grouped = [np.array([y_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    sessions_grouped = [np.array([sessions_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    ids_grouped = [np.array([ids_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    blocks_grouped = [np.array([blocks_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    channels_grouped = [np.array([channels_tmp[i] for i in ind[0]]) for ind in grouping_indices]

    X_seg, activities_seg, sessions_seg, ids_seg, blocks_seg, channels_seg = [], [], [], [], [], []
    
    # Transform into windows
    for in_channelgroup in range(0, len(X_grouped)):
        X_seg.append(windowsTransformer.fit_transform(X_grouped[in_channelgroup]))
        activities_seg.append(windowsTransformer.transform(activities_grouped[in_channelgroup], is_labels=True))
        sessions_seg.append(windowsTransformer.transform(sessions_grouped[in_channelgroup], is_labels=True))
        ids_seg.append(windowsTransformer.transform(ids_grouped[in_channelgroup], is_labels=True))
        blocks_seg.append(windowsTransformer.transform(blocks_grouped[in_channelgroup], is_labels=True))
        channels_seg.append(windowsTransformer.transform(channels_grouped[in_channelgroup], is_labels=True))
     
    # ------------------------------------------------------------
    ### Get kernel matrices with HybridBlocks
    matrices = []
    
    for in_channelgroup in range(len(X_seg)):
        actual_block_sizes = blocks_seg[in_channelgroup][0]
        if sum_blocks: 
            block_size = [actual_block_sizes[0+i*2] + actual_block_sizes[1+i*2] for i in range(exp_n_blocks)]
        else: 
            block_size = list(actual_block_sizes)
        assert len(actual_block_sizes) == exp_n_blocks, "Block number in data does not match expected block number."
        block_kernels = HybridBlocks(block_size=block_size,
                                 shrinkage=shrinkage,
                                 metrics=metrics)
        matrices.append(block_kernels.fit_transform(X_seg[in_channelgroup]))

    # ------------------------------------------------------------
    ### Project matrices into common space
    if use_missing_channels:
        print('Projecting matrices into common space')
        target_dim = np.min(np.unique(channels_tmp))
        for in_channelgroup in range(0, len(matrices)):
            if matrices[in_channelgroup][0].shape[0] != target_dim:
                matrices[in_channelgroup] = project_to_common_space(matrices[in_channelgroup], target_dim)
    X_common = np.concatenate(matrices)
    activities_common = np.concatenate(activities_seg)
    sessions_common = np.concatenate(sessions_seg)
    ids_common = np.concatenate(ids_seg)
    channels_common = np.concatenate(channels_seg)

    # ------------------------------------------------------------
    ### Demean matrices
    # Set group variables for demeaner
    if demean: 
        if clustering == 'full': 
            if demeaner_var == 'id-wise':
               groups = ids_common
            elif demeaner_var == 'session-wise':
                groups = sessions_common
            else: 
                demean = False
                print("Warning: will not demean because demeaning mode is unclear.")
        elif clustering == 'id-wise': 
            if demeaner_var == 'session-wise':
                groups = sessions_common
            elif demeaner_var == 'id-wise': 
                demean = False
                print("Warning: will not demean id-wise because clustering is id-wise.")
            else: 
                demean = False
                print("Warning: will not demean because demeaning mode is unclear.")
        elif clustering == 'session-wise': 
            demean = False
            print("Warning: will not demean because clustering is session-wise.")
    if demean: 
        demeaner = Demeaner(groups=groups,
                        activate=demean,
                        method=demeaner_method)
        matrices = demeaner.fit_transform(X_common)
    
    else:
        matrices = X_common

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
    sh_score_pipeline = riemannian_silhouette_score(matrices, classes)
    ch_score_pipeline = ch_score(matrices, classes)
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
    rand_score_id = adjusted_rand_score(classes, ids_common) if clustering == 'all' else np.nan
    rand_score_ses = adjusted_rand_score(classes, sessions_common) if clustering != 'session-wise' else np.nan
    rand_score_act = adjusted_rand_score(classes, activities_common)
    #rand_score_chs = adjusted_rand_score(classes, channels_common)

    # ------------------------------------------------------------
    ### PCA 
    mean_matrix = mean_riemann(matrices)
    X_tangent = tangent_space(matrices, mean_matrix)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tangent)

    # Plot
    if plot and not grid_search: 
        # With classes as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(classes): 
            plt.scatter(X_pca[classes == label, 0], X_pca[classes == label, 1], label=f"Class {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for ID {id}, clustering")
        plt.legend()
        plt.show()
        
        # With n_channels as labels
        if use_missing_channels:
            plt.figure(figsize=(6, 5))
            for label in np.unique(channels_common): 
                plt.scatter(X_pca[channels_common == label, 0], X_pca[channels_common == label, 1], label=f"n_channels: {label}", alpha=0.8)
            plt.xlabel("PC1")
            plt.xlabel("PC2")
            plt.title(f"Tangent Space PCA projection for ID {id}, channel numbers")
            plt.legend()
            plt.show()

        # With session as labels
        if clustering != 'session-wise':
            plt.figure(figsize=(6, 5))
            for label in np.unique(sessions_common): 
                plt.scatter(X_pca[sessions_common == label, 0], X_pca[sessions_common == label, 1], label=f"Session: {label}", alpha=0.8)
            plt.xlabel("PC1")
            plt.xlabel("PC2")
            plt.title(f"Tangent Space PCA projection for ID {id}, sessions")
            plt.legend()
            plt.show()

        # With activity as labels
        plt.figure(figsize=(6, 5))
        for label in np.unique(activities_common)[::-1]: 
            plt.scatter(X_pca[activities_common == label, 0], X_pca[activities_common == label, 1], label=f"Activity: {label}", alpha=0.8)
        plt.xlabel("PC1")
        plt.xlabel("PC2")
        plt.title(f"Tangent Space PCA projection for ID {id}, activities")
        plt.legend()
        plt.show()
    
    results = [matrices, cluster_means, classes, activities_common, sessions_common, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, len(sessions_tmp)]
    return results

def get_counts(strings):
    a = b = c = d = 0
    participants = ['target', 'partner'] 
    chromophore = ['hbr', 'hbo']
    level1 = chromophore if type_of_data in {"four_blocks", "four_blocks_session"} else participants
    level2 = participants if type_of_data in {"four_blocks", "four_blocks_session"} else chromophore

    for s in strings:
        if s.find(level1[0]) != -1 or type_of_data in {"one_brain", "one_brain_session"}: 
            if s.find(level2[0]) != -1:
                a += 1
            elif s.find(level2[1]) != -1:
                b += 1
        elif s.find(level1[1]) != -1:
            if s.find(level2[0]) != -1:
                c += 1
            elif s.find(level2[1]) != -1:
                d += 1

    if type_of_data in {"one_brain", "one_brain_session"}:
        return a, b
    else: 
        return a, b, c, d
    
# ------------------------------------------------------------
### Load data. Do not change this section. 

# Load the dataset
npz_data = np.load(f"./data/ts_{type_of_data}_fb{which_freq_bands}.npz")
X = []
for array in list(npz_data.files):
    X.append(npz_data[array])
doc = pd.read_csv(f"./data/doc_{type_of_data}.csv", index_col = 0)
ids = np.array(doc['0'])
sessions = np.array(doc['1'])
conditions = [
    (doc['2'] == 0),
    (doc['2'] == 1) | (doc['2'] == 2),
    (doc['2'] == 3)]
choices = ['alone', 'collab', 'diverse']
y = np.select(conditions, choices, default='unknown')
npz_channels = np.load(f"./data/channels_{type_of_data}.npz")
channels = []
for array in list(npz_channels.files):
    channels.append(npz_channels[array])

# make variable for chosen ids
chosen_ids = np.unique(ids) if which_id == 'all' else [which_id]

# choose only drawing alone and collaborative drawing
X = [i for idx, i in enumerate(X) if y[idx] != 'diverse']
drawing_indices = np.where(y != 'diverse')[0]
y = y[drawing_indices]
ids = ids[drawing_indices]
sessions = sessions[drawing_indices]
channels = [channels[i] for i in drawing_indices]
blocks = []
for i in range(len(channels)):
    blocks.append(get_counts(channels[i]))
blocks = np.array(blocks)
n_channels = np.array([len(channels[i]) for i in range(0, len(channels))])

# channels and block size
if type_of_data in {"one_brain", "one_brain_session"}:
    exp_n_channels = 8
    exp_n_blocks = 2 if exp_block_size == 4 else 1
else:
    exp_n_channels = 16
    exp_n_blocks = 4 if exp_block_size == 4 else 2
sum_blocks = exp_block_size == 8
if exp_block_size not in {4, 8}:
    raise ValueError('Unknown expected block size. Choose from 4, 8.')

# check demeaner
if demean and (demeaner_method not in {'log-euclidean', 'tangent', 'projection', 'airm'}) or (demeaner_var not in {'id-wise', 'session-wise'}):
    raise ValueError("No method for demeaner set. Choose from 'log-euclidean', 'tangent', 'projection', 'airm'")

# make dict
freq_bands = [[0.015, 0.4], [0.1, 0.2], [0.03, 0.1], [0.02, 0.03]]
if grid_search:
    dict = {
        "type_of_data": type_of_data,
        "block_size": exp_block_size,
        "l_freq": freq_bands[which_freq_bands][0],
        "h_freq": freq_bands[which_freq_bands][1],
        "use_missing_channels": use_missing_channels,
        "clustering": clustering,
        "which_id": which_id,
        "which_session": which_session,
        "demean": demean,
        "demeaner_method": demeaner_method,
        "demeaner_var": demeaner_var,
        "shrinkage": params_shrinkage,
        "metrics": params_kernel,
        "n_clusters": params_n_clusters,
        "s_freq": upsampling_freq,
        "window_size": window_length,
        "step_size": step_length,
    }
else: 
    dict = {
        "type_of_data": type_of_data,
        "block_size": exp_block_size,
        "l_freq": freq_bands[which_freq_bands][0],
        "h_freq": freq_bands[which_freq_bands][1],
        "use_missing_channels": use_missing_channels,
        "clustering": clustering,
        "which_id": which_id, 
        "which_session": which_session,
        "demean": demean,
        "demeaner_method": demeaner_method,
        "demeaner_var": demeaner_var,
        "shrinkage": shrinkage,
        "metrics": metrics,
        "n_clusters": n_clusters,
        "s_freq": upsampling_freq,
        "window_size": window_length,
        "step_size": step_length,
    }

# ------------------------------------------------------------
### Run pipeline. Do not change this section. 

if not grid_search:
    scores = []
    if clustering == 'full':
        results = pipeline(
                X, y, id=np.nan, session=np.nan, 
                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                plot=plot, window_length=window_length, step_length=step_length, 
                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
        scores.append(
                ['all', 'all', results[5], results[6], results[7], results[8], 
                    results[9], results[10], results[11], results[12], results[13]]
            )
        matrices, cluster_means, classes = results[0], results[1], results[2]
    elif clustering == 'id-wise':
        for id in chosen_ids: 
            results = pipeline(
                X, y, id=id, session=np.nan, 
                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                plot=plot, window_length=window_length, step_length=step_length, 
                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
            scores.append(
                [id, 'all', results[5], results[6], results[7], results[8], 
                    results[9], results[10], results[11], results[12], results[13]]
            )
            matrices, cluster_means, classes = results[0], results[1], results[2]
    elif clustering == 'session-wise':
        for id in chosen_ids:
            # make variable for chosen sessions
            chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]

            for session in chosen_sessions: 
                results = pipeline(
                    X, y, id=id, session=session, 
                    demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                    plot=plot, window_length=window_length, step_length=step_length, 
                    shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                scores.append(
                    [id, session, results[5], results[6], results[7], results[8], 
                        results[9], results[10], results[11], results[12], results[13]]
                ) 
                matrices, cluster_means, classes = results[0], results[1], results[2]
    scores = pd.DataFrame(scores, columns = [
        'ID', 'Session', 'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio',
        'RandScoreActivities', 'RandScoreSessions', 'RandScoreIds', 'nActivities']
    )   

# ------------------------------------------------------------
### Run grid search. Do not change this section. 

if grid_search:
    # Compute grid search parameters from inputs
    comb_shrinkage = product(params_shrinkage, repeat = exp_n_blocks)
    params_shrinkage_combinations = [list(x) for x in comb_shrinkage]
    comb_kernel = product(params_kernel, repeat = exp_n_blocks)
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
                        results = pipeline(
                        X, y, id=np.nan, session=np.nan, 
                        demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                        plot=plot, window_length=window_length, step_length=step_length,
                        shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                    except ValueError as e:
                        print(f"Skipping due to error: {e}")
                        continue
                    scores.append(
                        ['all', 'all', window_length, shrinkage, kernel, n_clusters, 
                            results[5], results[6], results[7], results[8], 
                            results[9], results[10], results[11], results[12], results[13]]
                    )
                elif clustering == 'id-wise':
                    for id in chosen_ids:
                        i += 1
                        print(f"Iteration {i}, parameters: ID {id}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                        try:
                            results = pipeline(
                                X, y, id=id, session=np.nan, 
                                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                plot=plot, window_length=window_length, step_length=step_length,
                                shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                        except ValueError as e:
                            print(f"Skipping due to error: {e}")  
                            continue
                        scores.append(
                            [id, 'all', window_length, shrinkage, kernel, n_clusters, 
                                results[5], results[6], results[7], results[8], 
                                results[9], results[10], results[11], results[12], results[13]])
                elif clustering == 'session-wise':
                    for id in chosen_ids:
                        chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]
                        for session in chosen_sessions:
                            i += 1
                            print(f"Iteration {i}, parameters: ID {id}, session {session}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                            try:
                                results = pipeline(
                                    X, y, id=id, session=session, 
                                    demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                    plot=plot, window_length=window_length, step_length=step_length,
                                    shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                            except ValueError as e:
                                print(f"Skipping due to error: {e}")  # Optional: print the error message
                                continue
                            except AssertionError as e:
                                print(f"Skipping due to error: {e}")  # Optional: print the error message
                                continue
                            scores.append(
                                [id, session, window_length, shrinkage, kernel, n_clusters, 
                                    results[5], results[6], results[7], results[8], 
                                    results[9], results[10], results[11], results[12], results[13]])
    scores = pd.DataFrame(scores, columns=['ID', 'Session', 'WindowLength', 'Shrinkage', 'Kernel', 'nClusters', 
                                       'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio'
                                       'RandScoreActivities', 'RandScoreSessions', 'RandScoreIds', 'nSessions'])


# ------------------------------------------------------------
### Save results. Do not change this section. 
print("saving results")
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
scores.to_csv(Path(outpath, f"parameter_space_scores_{type_of_data}_{timestamp}.csv"), index=False)
if not grid_search: 
    np.save(Path(outpath, f"matrices_{type_of_data}_{timestamp}.npy"), matrices)
    np.save(Path(outpath, f"cluster_means_{type_of_data}_{timestamp}.npy"), cluster_means)
    np.save(Path(outpath, f"classes_{type_of_data}_{timestamp}.npy"), classes)
json_object = json.dumps(dict, indent=4)
with open(Path(outpath, f"pipeline_description_{timestamp}.json"), "w") as outfile: 
    outfile.write(json_object)