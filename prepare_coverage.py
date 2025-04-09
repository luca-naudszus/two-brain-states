# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 11 March 2025

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

path = '/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/analysis'
type_of_data = "four_blocks"
ageDPFs = False
demean = "false"


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
adpfs = "ageDPFs" if ageDPFs else "usual"
outpath = Path(path) / "results" / adpfs / f"demean-{demean}"

fn = sorted(list(outpath.glob(f"results_table_{type_of_data}_*")))[-1]
results_table = pd.read_csv(fn)

# ------------------------------------------------------------
# Calculate coverage
coverage_table, coverage_table_single, entropy_table, entropy_table_single = [], [], [], []
sorted_activities = sorted(set(results_table.activities))
if type_of_data == "one_brain": 
    true_dyads = pd.read_csv(Path(path) / "data" / "dyadList.csv")
    dyads = pseudodyads(true_dyads) 
    for i, row in dyads.iterrows(): 
        #TODO: Why do some IDs only appear in targetID and not in partnerID?
        targetID, partnerID, dyadType, dyadID, group = row['pID1'], row['pID2'], row['dyadType'], row['dyadID'], row['group']
        for session in range(6):
            for activity in sorted_activities:
                session_activity_df = results_table[
                    (results_table.activities == activity) & 
                    (results_table.sessions == session)
                ]
                target = session_activity_df.loc[session_activity_df.ids == targetID, "classes"]
                partner = session_activity_df.loc[session_activity_df.ids == partnerID, "classes"]
                if not (target.empty or partner.empty): 
                    classes = np.stack((target + 1, partner + 1), axis=1)
                    n = len(classes)
                    classes_counter = make_counter(classes, is_tuple=True)
                    pk = []
                    for c1, c2 in product(np.unique(classes), repeat=2): 
                        ### coverage
                        coverage = np.sum((classes[:, 0] == c1) & (classes[:, 1] == c2))
                        if coverage != 0: 
                            ### stability (mean duration a given microstate remains stable, i.e. occurs consecutively)
                            stability = np.mean(classes_counter[(c1, c2)])
                            ### occurrence (mean number of times a microstate occurred during a one second period)
                            occurrence = len(classes_counter[(c1, c2)])
                        else: 
                            stability = np.nan
                            occurrence = 0
                        pk += [coverage/n]
                        coverage_table.append([
                            coverage, stability, occurrence, dyadID, dyadType, group, session + 1, activity,
                                f"{c1}_{c2}", n           
                            ])
                    entropy = sp.stats.entropy(pk)
                    entropy_table.append([
                        entropy, dyadID, dyadType, group, session + 1, activity
                    ])
        if dyadType: 
            for session in range(6):
                for activity in sorted_activities:
                    target = results_table.classes[
                                (results_table.activities == activity) & 
                                (results_table.sessions == session) & 
                                (results_table.ids == targetID)]
                    target_counter = make_counter(target)
                    partner = results_table.classes[
                                (results_table.activities == activity) & 
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
                                stability_ct = np.mean(target_counter[class_t])
                                occurrence_ct = len(target_counter[class_t])
                            else: 
                                stability_ct = np.nan
                                occurrence_ct = 0
                            pk_ct += [cov_ct/n_t]
                            coverage_table_single.append(
                                [cov_ct, stability_ct, occurrence_ct, targetID, group_t, session + 1, activity, class_t + 1, n_t]
                            )
                        entropy_ct = sp.stats.entropy(pk_ct)
                        entropy_table_single.append([
                            entropy_ct, targetID, group_t, session + 1, activity
                        ])
                    for class_p in np.unique(partner):
                        n_p = len(partner)
                        pk_cp = []
                        if n_p != 0:
                            cov_cp = np.sum(partner == class_p)
                            group_p = "young" if partnerID < 300 else "old"
                            if cov_cp != 0:
                                stability_cp = np.mean(partner_counter[class_p])
                                occurrence_cp = len(partner_counter[class_p])
                            else: 
                                stability_cp = np.nan
                                occurrence_cp = 0
                            pk_cp = [cov_cp/n_p]
                            coverage_table_single.append([
                                cov_cp, stability_cp, occurrence_cp, partnerID, group_p, session + 1, activity, class_p + 1, n_p
                            ])
                        entropy_cp = sp.stats.entropy(pk_cp)
                        entropy_table_single.append([
                            entropy_cp, partnerID, group_p, session + 1, activity
                        ])

else: 
    for id in sorted(set(results_table.ids)):
        #TODO: Make pseudo dyads have a group. 
        if (len(str(id)) == 4): 
            dyadType = True
            if str(id).startswith("1"): 
                group = "Same gen"
            else:
                group = "Intergen"
        else:
            dyadType = False
            group = "None"
        for session in sorted(set(results_table.sessions)):
            for activity in sorted_activities:
                n_total = len(results_table[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                if n_total > 0:
                    classes = results_table.classes[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)]
                    classes_counter = make_counter(classes)
                    pk = []
                    for cluster in sorted(set(classes)):          
                        coverage = len(results_table[
                            (results_table.classes == cluster) & 
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                        if coverage != 0:
                            stability = np.mean(classes_counter[cluster])
                            occurrence = len(classes_counter[cluster])
                        else: 
                            stability = np.nan
                            occurrence = 0
                        pk += [coverage/n_total]
                        coverage_table.append(
                            [coverage, stability, occurrence, id, dyadType, group, session + 1, activity, cluster + 1, n_total]
                        )
                    entropy = sp.stats.entropy(pk)
                    entropy_table.append([
                        coverage, id, dyadType, group, session + 1, activity       
                    ])

coverage_table = pd.DataFrame(coverage_table, 
             columns = ['coverage', 'stability', 'occurrence', 'id', 'dyadType', 'group', 'session', 'activity', 'cluster', 'n'])
entropy_table = pd.DataFrame(entropy_table,
                             columns = ['entropy', 'id', 'dyadType', 'group', 'session', 'activity'])
if type_of_data == "one_brain":
    coverage_table_single = pd.DataFrame(coverage_table_single,
                                         columns= ['coverage', 'stability', 'occurrence', 'id', 'group', 'session', 'activity', 'cluster', 'n'])
    entropy_table_single = pd.DataFrame(entropy_table_single,
                                         columns= ['entropy', 'id', 'group', 'session', 'activity'])

# ------------------------------------------------------------
### Save results. 

if type_of_data == "one_brain":
    coverage_table.to_csv((outpath / f"coverage_table_{type_of_data}_shared.csv"), index=False)
    coverage_table_single.to_csv((outpath / f"coverage_table_{type_of_data}_single.csv"), index=False)
    entropy_table.to_csv((outpath / f"entropy_table_{type_of_data}_shared.csv"), index=False)
    entropy_table_single.to_csv((outpath / f"entropy_table_{type_of_data}_single.csv"), index=False)
else: 
    coverage_table.to_csv((outpath / f"coverage_table_{type_of_data}.csv"), index=False)
    entropy_table.to_csv((outpath / f"entropy_table_{type_of_data}.csv"), index=False)
    
