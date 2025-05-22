# Extracting features from brain state sequence time series

#**Author:** Luca A. Naudszus
#**Date:** 11 March 2025
#**Affiliation:** Social Brain Sciences Lab, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch

# ------------------------------------------------------------
# Import packages and custom functions

from collections import defaultdict
from pathlib import Path
#---
from itertools import product
import numpy as np
import pandas as pd
import scipy as sp
#---
from riemannianKMeans import pseudodyads

# ------------------------------------------------------------
# Set variables
path = '/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/analysis/data/time-series-features'
type_of_data = "one-brain-3" # "one-brain-3", "one-brain-8", "four-blocks-3" or "four-blocks-7"

# ------------------------------------------------------------
# Custom functions

def make_counter(arr, is_tuple=False): 
    if is_tuple:
        classes_series = [tuple(row) for row in arr]
    else: 
        classes_series = [row for row in arr]
    runs = defaultdict(list)
    prev = None
    count = 0
    for item in classes_series + [None]: 
        if item == prev:
            count += 1
        else: 
            if prev is not None: 
                runs[prev].append(count)
            count = 1
            prev = item
    return runs

# ------------------------------------------------------------
# Load Data
datapath = Path(path) / "clustering" / type_of_data
outpath = Path(path) / type_of_data
if not outpath.is_dir():
    outpath.mkdir()
fn = sorted(list(datapath.glob(f"results-table_{type_of_data}_*")))[-1]
results_table = pd.read_csv(fn)
results_table.columns = ["classes", "tasks", "sessions", "ids"]

# ------------------------------------------------------------
# Calculate coverage
feature_table, feature_table_single, entropy_table, entropy_table_single = [], [], [], []
sorted_tasks = sorted(set(results_table.tasks))
if "one-brain" in type_of_data: 
    true_dyads = pd.read_csv(Path(path) / "clustering" / "fNIRS_prepared" / "additional_information" / "dyadList.csv")
    dyads = pseudodyads(true_dyads) 
    for i, row in dyads.iterrows(): 
        targetID, partnerID, dyad_type, dyadID, group = row['pID1'], row['pID2'], row['dyad_type'], row['dyadID'], row['group']
        dyad_type_formatted = "Real" if dyad_type else "Pseudo"
        for session in range(6):
            for task in sorted_tasks:
                session_task_df = results_table[
                    (results_table.tasks == task) & 
                    (results_table.sessions == session)
                ]
                target = session_task_df.loc[session_task_df.ids == targetID, "classes"]
                partner = session_task_df.loc[session_task_df.ids == partnerID, "classes"]
                if not (target.empty or partner.empty): 
                    classes = np.stack((target + 1, partner + 1), axis=1)
                    n = len(classes)
                    classes_counter = make_counter(classes, is_tuple=True)
                    pk = []
                    for c1, c2 in product(np.unique(classes), repeat=2): 
                        ### coverage
                        coverage = np.sum((classes[:, 0] == c1) & (classes[:, 1] == c2))
                        if coverage != 0: 
                            ### duration (mean duration a given microstate remains stable, i.e. occurs consecutively)
                            duration = np.mean(classes_counter[(c1, c2)])
                            ### occurrence (mean number of times a microstate occurred during a one second period)
                            occurrence = len(classes_counter[(c1, c2)])
                        else: 
                            duration = np.nan
                            occurrence = 0
                        pk += [coverage/n]
                        feature_table.append([
                            coverage, duration, occurrence, dyadID, dyad_type_formatted, group, session + 1, task,
                                f"{c1}_{c2}", n           
                            ])
                    entropy = sp.stats.entropy(pk)
                    entropy_table.append([
                        entropy, dyadID, dyad_type_formatted, group, session + 1, task
                    ])
        if dyad_type: 
            for session in range(6):
                for task in sorted_tasks:
                    target = results_table.classes[
                                (results_table.tasks == task) & 
                                (results_table.sessions == session) & 
                                (results_table.ids == targetID)]
                    target_counter = make_counter(target)
                    partner = results_table.classes[
                                (results_table.tasks == task) & 
                                (results_table.sessions == session) & 
                                (results_table.ids == partnerID)]
                    partner_counter = make_counter(partner)
                    for class_t in np.unique(target):
                        n_t = len(target)
                        pk_ct = []
                        if n_t != 0:
                            cov_ct = np.sum(target == class_t) 
                            group_t = "young" if targetID < 300 else "old"
                            if cov_ct != 0:
                                duration_ct = np.mean(target_counter[class_t])
                                occurrence_ct = len(target_counter[class_t])
                            else: 
                                duration_ct = np.nan
                                occurrence_ct = 0
                            pk_ct += [cov_ct/n_t]
                            feature_table_single.append(
                                [cov_ct, duration_ct, occurrence_ct, targetID, group_t, session + 1, task, class_t + 1, n_t]
                            )
                        entropy_ct = sp.stats.entropy(pk_ct)
                        entropy_table_single.append([
                            entropy_ct, targetID, group_t, session + 1, task
                        ])
                    for class_p in np.unique(partner):
                        n_p = len(partner)
                        pk_cp = []
                        if n_p != 0:
                            cov_cp = np.sum(partner == class_p)
                            group_p = "young" if partnerID < 300 else "old"
                            if cov_cp != 0:
                                duration_cp = np.mean(partner_counter[class_p])
                                occurrence_cp = len(partner_counter[class_p])
                            else: 
                                duration_cp = np.nan
                                occurrence_cp = 0
                            pk_cp = [cov_cp/n_p]
                            feature_table_single.append([
                                cov_cp, duration_cp, occurrence_cp, partnerID, group_p, session + 1, task, class_p + 1, n_p
                            ])
                        entropy_cp = sp.stats.entropy(pk_cp)
                        entropy_table_single.append([
                            entropy_cp, partnerID, group_p, session + 1, task
                        ])

elif "four-blocks" in type_of_data: 
    for id in sorted(set(results_table.ids)):
        if (len(str(id)) == 4): 
            dyad_type = "Real"
            group = "Same gen" if str(id).startswith("1") else "Intergen"

        else: 
            dyad_type = "Pseudo"
            group = "Same gen" if (id[:3] < 300) and (id[-3:] < 300) else "Intergen"

        for session in sorted(set(results_table.sessions)):
            for task in sorted_tasks:
                n_total = len(results_table[
                            (results_table.tasks == task) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                if n_total > 0:
                    classes = results_table.classes[
                            (results_table.tasks == task) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)]
                    classes_counter = make_counter(classes)
                    pk = []
                    for state in sorted(set(classes)):          
                        coverage = len(results_table[
                            (results_table.classes == state) & 
                            (results_table.tasks == task) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                        if coverage != 0:
                            duration = np.mean(classes_counter[state])
                            occurrence = len(classes_counter[state])
                        else: 
                            duration = np.nan
                            occurrence = 0
                        pk += [coverage/n_total]
                        feature_table.append(
                            [coverage, duration, occurrence, id, dyad_type, group, session + 1, task, state + 1, n_total]
                        )
                    entropy = sp.stats.entropy(pk)
                    entropy_table.append([
                        coverage, id, dyad_type, group, session + 1, task       
                    ])

feature_table = pd.DataFrame(feature_table, 
             columns = ['coverage', 'duration', 'occurrence', 'dyad', 'dyad_type', 'group', 'session', 'task', 'state', 'n'])
entropy_table = pd.DataFrame(entropy_table,
                             columns = ['entropy', 'dyad', 'dyad_type', 'group', 'session', 'task'])
if "one-brain" in type_of_data: 
    feature_table_single = pd.DataFrame(feature_table_single,
                                         columns= ['coverage', 'duration', 'occurrence', 'id', 'group', 'session', 'task', 'state', 'n'])
    entropy_table_single = pd.DataFrame(entropy_table_single,
                                         columns= ['entropy', 'id', 'group', 'session', 'task'])

# ------------------------------------------------------------
### Save results. 

if "one-brain" in type_of_data: 
    feature_table.to_csv((outpath / f"feature-table_{type_of_data}_combined.csv"), index=False)
    feature_table_single.to_csv((outpath / f"feature-table_{type_of_data}_single.csv"), index=False)
    entropy_table.to_csv((outpath / f"entropy-table_{type_of_data}_combined.csv"), index=False)
    entropy_table_single.to_csv((outpath / f"entropy-table_{type_of_data}_single.csv"), index=False)
else: 
    feature_table.to_csv((outpath / f"feature-table_{type_of_data}.csv"), index=False)
    entropy_table.to_csv((outpath / f"entropy-table_{type_of_data}.csv"), index=False)
    
