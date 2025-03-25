# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 11 March 2025

# ------------------------------------------------------------
# Import packages and custom functions

import os
from pathlib import Path
#---
from itertools import product
import numpy as np
import pandas as pd
#---
from riemannianKMeans import pseudodyads

# ------------------------------------------------------------
# Set variables

os.chdir('C://Users//SBS_T//Documents//Luca')
type_of_data = "one_brain"
ageDPFs = True

# ------------------------------------------------------------
# Load Data
if ageDPFs: 
    path = Path('results', 'ageDPFs')
else: 
    path = Path('results')
fn = sorted(list(path.glob(f"results_table_{type_of_data}_*")))[-1]
results_table = pd.read_csv(fn)

# ------------------------------------------------------------
# Calculate coverage
coverage_table, coverage_table_single = [], []
if type_of_data == "one_brain": 
    true_dyads = pd.read_csv(Path("data") / "dyadList.csv")
    dyads = pseudodyads(true_dyads) 
    for i, row in dyads.iterrows(): 
        #TODO: Why do some IDs only appear in targetID and not in partnerID?
        targetID, partnerID, dyadType, dyadID, group = row['pID1'], row['pID2'], row['dyadType'], row['dyadID'], row['group'], 
        for session in range(6):
            for activity in sorted(set(results_table.activities)):
                target = results_table.classes[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == targetID)]
                partner = results_table.classes[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == partnerID)]
                if len(target) != 0 and len(partner) != 0: 
                    classes = np.stack((target + 1, partner + 1), axis=1)
                    for class_combination in product(np.unique(classes), repeat=2):
                        n = len(target)
                        coverage = np.sum((classes[:,0] == class_combination[0]) & (classes[:,1] == class_combination[1])) 
                        coverage_table.append(
                        [coverage, dyadID, dyadType, group, session + 1, activity, str(class_combination[0]) + "_" + str(class_combination[1]), n]
                        )
        if dyadType: 
            for session in range(6):
                for activity in sorted(set(results_table.activities)):
                    target = results_table.classes[
                                (results_table.activities == activity) & 
                                (results_table.sessions == session) & 
                                (results_table.ids == targetID)]
                    partner = results_table.classes[
                                (results_table.activities == activity) & 
                                (results_table.sessions == session) & 
                                (results_table.ids == partnerID)]
                    for class_t in np.unique(target):
                        n_t = len(target)
                        if n_t != 0:
                            cov_ct = np.sum(target == class_t) 
                            group_t = "young" if targetID < 300 else "old"
                            coverage_table_single.append(
                                [cov_ct, targetID, group_t, session + 1, activity, class_t + 1, n_t]
                            )
                    for class_p in np.unique(partner):
                        n_p = len(partner)
                        if n_p != 0:
                            cov_cp = np.sum(partner == class_p)
                            group_p = "young" if partnerID < 300 else "old"
                            coverage_table_single.append(
                                [cov_cp, partnerID, group_p, session + 1, activity, class_p + 1, n_p]
                            )

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
            for activity in sorted(set(results_table.activities)):
                n_total = len(results_table[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                if n_total > 0:
                    for cluster in sorted(set(results_table.classes)):          
                        coverage = len(results_table[
                            (results_table.classes == cluster) & 
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                        coverage_table.append(
                        [coverage, id, dyadType, group, session + 1, activity, cluster + 1, n_total]
                        )

coverage_table = pd.DataFrame(coverage_table, 
             columns = ['coverage', 'id', 'dyadType', 'group', 'session', 'activity', 'cluster', 'n'])
if type_of_data == "one_brain":
    coverage_table_single = pd.DataFrame(coverage_table_single,
                                         columns= ['coverage', 'id', 'group', 'session', 'activity', 'cluster', 'n'])

# ------------------------------------------------------------
### Save results. 

coverage_table.to_csv((path / f"coverage_table_{type_of_data}.csv"), index=False)
if type_of_data == "one_brain":
    coverage_table_single.to_csv((path / f"coverage_table_{type_of_data}_single.csv"), index=False)
    
