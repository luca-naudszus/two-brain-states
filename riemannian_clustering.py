# Riemannian k-Means clustering for fNIRS brain data

#**Author:** Luca A. Naudszus
#**Date:** 20 February 2025
#**Affiliation:** Social Brain Sciences Lab, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch

# ------------------------------------------------------------
# Import packages and custom functions

from datetime import datetime
import json
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
    plot_clustering,
    project_to_common_space, 
    riemannian_davies_bouldin, 
    riemannian_silhouette_score, 
    riemannian_variance
)

# ------------------------------------------------------------
# Set path
path = '/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/analysis/data/time-series-features/clustering'

# ------------------------------------------------------------
# Set arguments. Change only variables in this section of the script. 

# which type of data are we interested in?
type_of_data = "one-brain"
# one-brain, two-blocks, four-blocks: channel-wise z-scoring
# one-brain_session, etc.: channel- and session-wise z-scoring
exp_block_size = 4
which_freq_bands = 0 # Choose from 0 (0.01 to 0.4), 1 (0.1 to 0.2), 2 (0.03 to 0.1), 3 (0.02 to 0.03). 
ageDPFs = False

# do we want to use pseudo dyads?
pseudo_dyads = False
# True has excessive memory usage and 
# cannot run on a standard machine at the moment. 
# True is invalid for type_of_data == "one-brain", 
# pseudo dyads are created later in that case. 

# do we want to use data with missing channels?
use_missing_channels = False
# if so, data from all sessions are projected into a common space

# how do we want to cluster?
# Choose from 'full', 'id-wise', 'session-wise'
clustering = 'full'
# Which dyad/participant do we want to look at? (only for id-wise and session-wise clustering)
which_id = 'all' # set which_id = 'all' for all dyads/participants
# Which session do we want to look at? (only for session-wise clustering)
which_session = 'all' # set which_session = 'all' for all sessions

# should the matrices be demeaned? 
demean = False
# if so, within-id or within-session?
demeaner_var = 'session-wise' # 'none', 'id-wise', 'session-wise'
# if so, which method?
demeaner_method = 'airm' # 'log-euclidean', 'projection', or 'airm'
#TODO: Projection is a quick and dirty solution which sometimes encounters errors. 
# Log-euclidean is the second-fastest and the fastest among the two meaningful implementations. 
# AIRM is slower, but more accurate (respects the curvature of the SPD manifold). 

# do we want to do a single run or a grid search? (False = single run, True = grid search)
grid_search = False
## in case of False, define hyperparameters below
## in case of True, define parameter space below

# are we interested in the plot? (True/False, overridden in case of grid search: no plot)
plot = True

# hyperparameters (overridden in case of grid search)
shrinkage = 0.1 # shrinkage value
metrics = 'rbf' # kernel function
n_clusters = 5 # number of clusters for k-means

# parameter space for grid search
params_shrinkage = [0, 0.01 ,0.1]
params_kernel = ['cov', 'rbf', 'lwf', 'tyl', 'corr']
params_n_clusters = [3] # sys.argv[1] for usage on Euler

# information on data
upsampling_freq = 5 # frequency to which the data have been upsampled
window_length = 15 # length of windows in s
step_length = 1 # steps 

# define global settings
n_jobs = -1 # use all available cores
random_state = 42 # random state for reproducibility
n_init = 10 # number of initializations for kMeans
max_iter = 5 # maximum number of iterations for kMeans

# ------------------------------------------------------------
# Define custom functions. Do not change this section. 

def pipeline(X, y, id, session, 
             clustering, sessions, ids, blocks, n_channels, 
             demean, demeaner_var, demeaner_method, plot, 
             window_length, step_length, 
             shrinkage, metrics, n_clusters): 
    
    # ------------------------------------------------------------
    ### Get data
    if clustering == 'full':
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = X, y, sessions, ids, blocks, n_channels
        print(f"Data loaded: {len(X_tmp)} trials")
    elif clustering == 'id-wise':
        indices = np.where(ids == id)[0]
        if len(indices) == 0:
            raise ValueError(f"ID {id} and session {session} not found in dataset.")
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [blocks[i] for i in indices], [n_channels[i] for i in indices]
        print(f"Data loaded for id {id}: {len(X_tmp)} trials")
    elif clustering == 'session-wise':
        indices = np.where((ids == id) & (sessions == session))[0]
        if len(indices) == 0:
            raise ValueError(f"ID {id} and session {session} not found in dataset.")
        X_tmp, y_tmp, sessions_tmp, ids_tmp, blocks_tmp, channels_tmp = [X[i] for i in indices], [y[i] for i in indices], [sessions[i] for i in indices], [ids[i] for i in indices], [blocks[i] for i in indices], [n_channels[i] for i in indices]
        print(f"Data loaded for id {id} and session {session}: {len(X_tmp)} trials, {X_tmp[0].shape[0]} channels")
    
    # ------------------------------------------------------------
    ### Group by number of channels and segment into windows
    print("Preparing data")
    if use_missing_channels: 
        # Exclude data with too few channels
        set_n_channels = np.unique(channels_tmp)
        set_n_channels = set_n_channels[set_n_channels >= (exp_n_channels/2)]
        grouping_indices = [[np.where(np.array(channels_tmp) == channel_no)[0]] for channel_no in set_n_channels]
    else:
        grouping_indices = [np.where(np.array(channels_tmp) == exp_n_channels)]
    windowsTransformer = ListTimeSeriesWindowTransformer(
            window_size = upsampling_freq*window_length,
            step_size = upsampling_freq*step_length
        )
    
    # Group variables
    X_grouped = [[X_tmp[i] for i in ind[0]] for ind in grouping_indices]
    tasks_grouped = [np.array([y_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    sessions_grouped = [np.array([sessions_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    ids_grouped = [np.array([ids_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    blocks_grouped = [np.array([blocks_tmp[i] for i in ind[0]]) for ind in grouping_indices]
    channels_grouped = [np.array([channels_tmp[i] for i in ind[0]]) for ind in grouping_indices]

    X_seg, tasks_seg, sessions_seg, ids_seg, blocks_seg, channels_seg = [], [], [], [], [], []
    
    # Transform into windows
    for in_channelgroup in range(len(X_grouped)):
        X_seg.append(windowsTransformer.fit_transform(X_grouped[in_channelgroup]))
        tasks_seg.append(windowsTransformer.transform(tasks_grouped[in_channelgroup], is_labels=True))
        sessions_seg.append(windowsTransformer.transform(sessions_grouped[in_channelgroup], is_labels=True))
        ids_seg.append(windowsTransformer.transform(ids_grouped[in_channelgroup], is_labels=True))
        blocks_seg.append(windowsTransformer.transform(blocks_grouped[in_channelgroup], is_labels=True))
        channels_seg.append(windowsTransformer.transform(channels_grouped[in_channelgroup], is_labels=True))
     
    # ------------------------------------------------------------
    ### Get kernel matrices with HybridBlocks
    X_prepared = []
    
    for in_channelgroup in range(len(X_seg)):
        actual_block_sizes = blocks_seg[in_channelgroup][0]
        if sum_blocks: 
            block_size = [actual_block_sizes[0+i*2] + actual_block_sizes[1+i*2] for i in range(exp_n_blocks)]
        else: 
            block_size = list(actual_block_sizes)
        assert len(block_size) == exp_n_blocks, "Block number in data does not match expected block number."
        block_kernels = HybridBlocks(block_size=block_size,
                                 shrinkage=shrinkage,
                                 metrics=metrics)
        X_prepared.append(block_kernels.fit_transform(X_seg[in_channelgroup]))

    # ------------------------------------------------------------
    ### Project matrices into common space
    if use_missing_channels:
        print('Projecting matrices into common space')
        target_dim = np.min(set_n_channels)
        for in_channelgroup in range(len(X_prepared)):
            if X_prepared[in_channelgroup][0].shape[0] != target_dim:
                X_prepared[in_channelgroup] = project_to_common_space(X_prepared[in_channelgroup], target_dim)
    X_common = np.concatenate(X_prepared)
    tasks_common = np.concatenate(tasks_seg)
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
                groups = ["_".join(map(str, t)) for t in list(zip(ids_common, sessions_common))]
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
    if demean and (len(np.unique(channels_common)) > 1): 
        demeaner = Demeaner(groups=groups,
                        activate=demean,
                        method=demeaner_method)
        final_matrices = demeaner.fit_transform(X_common)
    
    else:
        final_matrices = X_common

    # ------------------------------------------------------------
    ### KMeans
    kmeans = RiemannianKMeans(n_jobs=n_jobs,
                            n_clusters=n_clusters,
                            n_init=n_init)
    classes = kmeans.fit_predict(final_matrices)
    cluster_means = kmeans.centroids()
    clusters = [final_matrices[classes == i] for i in range(n_clusters)]
    
    # ------------------------------------------------------------
    ### Clustering performance evaluation
    #sh_score_pipeline = riemannian_silhouette_score(final_matrices, classes)
    sh_score_pipeline = np.nan
    ch_score_pipeline = ch_score(final_matrices, classes)
    riem_var_pipeline = [riemannian_variance(clusters[i], cluster_means[i]) for i in range(n_clusters)]
    db_score_pipeline = riemannian_davies_bouldin(clusters, cluster_means)
    gdr_pipeline = geodesic_distance_ratio(clusters, cluster_means)
    print(f"Silhouette Score: {sh_score_pipeline}")
    print(f"Calinski-Harabasz Score: {ch_score_pipeline}")
    print(f"Riemannian Variance: {np.mean(riem_var_pipeline)}")
    print(f"Davies-Bouldin-Index: {db_score_pipeline}")
    print(f"Geodesic Distance Ratio: {gdr_pipeline}")

    # ------------------------------------------------------------
    ### Rand indices
    rand_score_id = adjusted_rand_score(classes, ids_common) if clustering == 'all' else np.nan
    rand_score_ses = adjusted_rand_score(classes, sessions_common) if clustering != 'session-wise' else np.nan
    rand_score_act = adjusted_rand_score(classes, tasks_common)
    # rand_score_chs = adjusted_rand_score(classes, channels_common) # adjust return results section for inclusion

    # ------------------------------------------------------------
    ### Plot clustering
    if plot and not grid_search: 
        plot_clustering(final_matrices, classes, "cluster")
        if use_missing_channels: 
            plot_clustering(final_matrices, channels_common, "n_channels")
        if clustering != 'session-wise':
            plot_clustering(final_matrices, sessions_common, "session")
        plot_clustering(final_matrices, tasks_common, "task")
    
    # ------------------------------------------------------------
    # Return results
    results = [final_matrices, cluster_means, classes, tasks_common, sessions_common, ids_common, sh_score_pipeline, ch_score_pipeline, riem_var_pipeline, db_score_pipeline, gdr_pipeline, rand_score_act, rand_score_ses, rand_score_id, len(sessions_tmp)]
    return results

def get_counts(strings):
    a = b = c = d = 0
    participants = ['target', 'partner'] 
    chromophore = ['hbr', 'hbo']
    level1 = chromophore if type_of_data in {"four-blocks", "four-blocks_session"} else participants
    level2 = participants if type_of_data in {"four-blocks", "four-blocks_session"} else chromophore

    for s in strings:
        if level1[0] in s or type_of_data in {"one-brain", "one-brain_session"}: 
            if level2[0] in s:
                a += 1
            elif level2[1] in s:
                b += 1
        elif level1[1] in s:
            if level2[0] in s:
                c += 1
            elif level2[1] in s:
                d += 1

    if type_of_data in {"one-brain", "one-brain_session"}:
        return a, b
    else: 
        return a, b, c, d
    
# ------------------------------------------------------------
### Load data. Do not change this section. 

# checks
if type_of_data == "one-brain" and pseudo_dyads: 
    pseudo_dyads = False
    raise ValueError("Set pseudo_dyads = False for one brain data. For one brain data, pseudo dyads are created in a later step.")
if type_of_data == "one-brain" and exp_block_size == 8: 
    print("Warning: one-brain data with block size 8 is unusual, check if this is what you want.")
if exp_block_size not in {4, 8}:
    raise ValueError('Unknown expected block size. Choose from 4, 8.')

if clustering not in {'full', 'id-wise', 'session-wise'}:
    raise ValueError(f"Unknown clustering type: {clustering}. Choose from 'full', 'id-wise', or 'session-wise'.")

if demean:
    if demeaner_method not in {'log-euclidean', 'tangent', 'projection', 'airm'}:
        raise ValueError("Invalid demeaner method. Choose from 'log-euclidean', 'tangent', 'projection', 'airm'")
    if demeaner_var not in {'id-wise', 'session-wise'}:
        raise ValueError("Invalid demeaner variable. Choose 'id-wise' or 'session-wise'")
else: 
    demeaner_method = "False"

# Make folder
datapath = Path(path) / "fNIRS_prepared" / type_of_data

# Load the dataset
pseudo = "true" if pseudo_dyads else "false"
npz_data = np.load(Path(datapath) / f"ts_{type_of_data}_fb-{which_freq_bands}_pseudo-{pseudo}.npz")
X = []
for array in list(npz_data.files):
    X.append(npz_data[array])
doc = pd.read_csv(Path(datapath) / f"doc_{type_of_data}_pseudo-{pseudo}.csv", index_col = 0)
ids = np.array(doc['id'])
sessions = np.array(doc['session'])
conditions = [
    (doc['task'] == 0),
    (doc['task'] == 1),
    (doc['task'] == 2),
    (doc['task'] == 3)]
choices = ['Alone', 'Together_1', 'Together_2', 'diverse']
y = np.select(conditions, choices, default='unknown')
npz_channels = np.load(Path(datapath) / f"channels_{type_of_data}_pseudo-{pseudo}.npz")
channels = []
for array in list(npz_channels.files):
    channels.append(npz_channels[array])

# make variable for chosen ids
chosen_ids = np.unique(ids) if which_id == 'all' else np.atleast_1d(which_id)

# choose only drawing alone and collaborative drawing
drawing_indices = np.where(y != 'diverse')[0]
X = [X[i] for i in drawing_indices]
y = y[drawing_indices]
ids = ids[drawing_indices]
sessions = sessions[drawing_indices]
channels = [channels[i] for i in drawing_indices]
blocks = np.array([get_counts(ch) for ch in channels])
n_channels = np.array([len(ch) for ch in channels])

# channels and block size
if type_of_data in {"one-brain", "one-brain_session"}:
    exp_n_channels = 8
    exp_n_blocks = 2 if exp_block_size == 4 else 1
else:
    exp_n_channels = 16
    exp_n_blocks = 4 if exp_block_size == 4 else 2
sum_blocks = exp_block_size == 8


# make dict
freq_bands = [[0.015, 0.4], [0.1, 0.2], [0.03, 0.1], [0.02, 0.03]]
pipeline_metadata = {
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
    "s_freq": upsampling_freq,
    "window_size": window_length,
    "step_size": step_length,
}

if grid_search:
    pipeline_metadata.update({
        "shrinkage": list(params_shrinkage),
        "metrics": list(params_kernel),
        "n_clusters": list(params_n_clusters),
    })
else:
    pipeline_metadata.update({
        "shrinkage": shrinkage,
        "metrics": metrics,
        "n_clusters": n_clusters,
    })

# ------------------------------------------------------------
### Run pipeline. Do not change this section. 

if not grid_search:
    scores = []
    if clustering == 'full':
        results = pipeline(
                X, y, id=np.nan, session=np.nan, 
                clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                plot=plot, window_length=window_length, step_length=step_length, 
                shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
        # Append id, sessions, SilhouetteCoefficient, CalinskiHarabaszScore, 
        # RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
        # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
        scores.append(
                ['all', 'all', results[6], results[7], results[8], results[9], results[10],
                    results[11], results[12], results[13], results[14]]
            )
        
    else:
        for id in chosen_ids: 
            if clustering == "session-wise":
                chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]

                for session in chosen_sessions: 
                    results = pipeline(
                        X, y, id=id, session=session, 
                        clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                        demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                        plot=plot, window_length=window_length, step_length=step_length, 
                        shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                    # Append id, sessions, SilhouetteCoefficient, CalinskiHarabaszScore, 
                    # RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
                    # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
                    scores.append(
                        [id, session, 
                            results[6], results[7], results[8], results[9], results[10],
                            results[11], results[12], results[13], results[14]]
                        ) 
            else: 
                results = pipeline(
                    X, y, id=id, session=np.nan, 
                    clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                    demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                    plot=plot, window_length=window_length, step_length=step_length, 
                    shrinkage=shrinkage, metrics=metrics, n_clusters=n_clusters)
                # Append id, sessions, SilhouetteCoefficient, CalinskiHarabaszScore, 
                # RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
                # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
                scores.append(
                    [id, 'all', 
                        results[6], results[7], results[8], results[9], results[10],
                        results[11], results[12], results[13], results[14]]
                    )
    scores = pd.DataFrame(scores, columns = [
        'ID', 'Session', 
        'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio',
        'RandscoreTasks', 'RandScoreSessions', 'RandScoreIds', 'nTasks']
        ) 
    matrices, cluster_means, classes, tasks, sessions, ids = results[0], results[1], results[2], results[3], results[4], results[5]
    results_table = pd.DataFrame(np.stack((classes, tasks, sessions, ids), axis = 1), 
                                 columns = ['classes', 'tasks', 'sessions', 'ids'])
      

# ------------------------------------------------------------
### Run grid search. Do not change this section. 

if grid_search:
    # ATTENTION: This script currently does not use the hybrid property of HybridBlocks. 
    # All hyperparameters are set to the same value in each iteration for all blocks. 
    # The commented-out lines below allow for separate shrinkage parameters and kernels per block.  
    # That option leads to much longer runtimes and is therefore avoided here. 

    # Compute grid search parameters from inputs
    comb_shrinkage = product(params_shrinkage, repeat = exp_n_blocks)
    params_shrinkage_combinations = [list(x) for x in comb_shrinkage]
    #params_shrinkage_combinations = params_shrinkage
    comb_kernel = product(params_kernel, repeat = exp_n_blocks)
    params_kernel_combinations = [list(x) for x in comb_kernel]
    #params_kernel_combinations = params_kernel
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
                        clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                        demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                        plot=plot, window_length=window_length, step_length=step_length,
                        shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                    except ValueError as e:
                        print(f"Skipping due to error: {e}")
                        continue
                    # Append id, sessions, window length, shrinkage, kernel, number of clusters,
                    # SilhouetteCoefficient, CalinskiHarabaszScore, RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
                    # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
                    scores.append(
                        ['all', 'all', window_length, shrinkage, kernel, n_clusters, 
                            results[6], results[7], results[8], results[9], results[10],
                            results[11], results[12], results[13], results[14]]
                    )
                else:
                    for id in chosen_ids:
                        if clustering == 'session-wise':
                            chosen_sessions = np.unique(sessions[ids == id]) if which_session == 'all' else [which_session]
                            for session in chosen_sessions:
                                i += 1
                                print(f"Iteration {i}, parameters: ID {id}, session {session}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                                try:
                                    results = pipeline(
                                        X, y, id=id, session=session, 
                                        clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                                        demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                        plot=plot, window_length=window_length, step_length=step_length,
                                        shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                                except ValueError as e:
                                    print(f"Skipping due to error: {e}")  # Optional: print the error message
                                    continue
                                except AssertionError as e:
                                    print(f"Skipping due to error: {e}")  # Optional: print the error message
                                    continue
                                # Append id, session, window length, shrinkage, kernel, number of clusters,
                                # SilhouetteCoefficient, CalinskiHarabaszScore, RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
                                # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
                                scores.append(
                                        [id, session, window_length, shrinkage, kernel, n_clusters, 
                                        results[6], results[7], results[8], results[9], results[10],
                                        results[11], results[12], results[13], results[14]]
                                )
                        else: 
                            i += 1
                            print(f"Iteration {i}, parameters: ID {id}, shrinkage {shrinkage}, kernel {kernel}, n_clusters {n_clusters}")
                            try:
                                results = pipeline(
                                    X, y, id=id, session=np.nan, 
                                    clustering=clustering, sessions=sessions, ids=ids, blocks=blocks, n_channels=n_channels,
                                    demean=demean, demeaner_var=demeaner_var, demeaner_method=demeaner_method,
                                    plot=plot, window_length=window_length, step_length=step_length,
                                    shrinkage=shrinkage, metrics=kernel, n_clusters=n_clusters)
                            except ValueError as e:
                                print(f"Skipping due to error: {e}")  
                                continue
                            # Append id, session, window length, shrinkage, kernel, number of clusters,
                            # SilhouetteCoefficient, CalinskiHarabaszScore, RiemannianVariance, DaviesBouldinIndex, GeodesicDistanceRatio,
                            # RandscoreTasks, RandScoreSessions, RandScoreIds, nTasks
                            scores.append(
                                        [id, 'all', window_length, shrinkage, kernel, n_clusters, 
                                        results[6], results[7], results[8], results[9], results[10],
                                        results[11], results[12], results[13], results[14]]
                            )
    scores = pd.DataFrame(scores, columns=['ID', 'Session', 'WindowLength', 'Shrinkage', 'Kernel', 'nClusters', 
                                       'SilhouetteCoefficient', 'CalinskiHarabaszScore', 'RiemannianVariance', 'DaviesBouldinIndex', 'GeodesicDistanceRatio',
                                       'RandscoreTasks', 'RandScoreSessions', 'RandScoreIds', 'nTasks'])


# ------------------------------------------------------------
### Save results. Do not change this section. 
print("saving results")
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

outpath = Path(path) / f"{type_of_data}-{n_clusters}"
if not outpath.is_dir():
    outpath.mkdir()

scores.to_csv(Path(outpath) / f"parameter-space-scores_{type_of_data}-{n_clusters}_{timestamp}.csv", index=False)
if not grid_search: 
    np.save(Path(outpath) / f"matrices_{type_of_data}-{n_clusters}_{timestamp}.npy", matrices)
    np.save(Path(outpath) / f"cluster-means_{type_of_data}-{n_clusters}_{timestamp}.npy", cluster_means)
    np.save(Path(outpath) / f"classes_{type_of_data}-{n_clusters}_{timestamp}.npy", classes)
    results_table.to_csv(Path(outpath) / f"results-table_{type_of_data}-{n_clusters}_{timestamp}.csv", index=False)
json_object = json.dumps(pipeline_metadata, indent=4)
with open(Path(outpath) / f"pipeline-description_{timestamp}.json", "w") as outfile: 
    outfile.write(json_object)
