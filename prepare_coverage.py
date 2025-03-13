# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 11 March 2025

# ------------------------------------------------------------
# Import packages and custom functions

import os
from pathlib import Path
#---
from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
#---
from riemannianKMeans import pseudodyads

# ------------------------------------------------------------
# Set variables

os.chdir('/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code')

type_of_data = "one_brain"

# ------------------------------------------------------------
# Load Data
path = Path('results')
fn = list(path.glob(f"results_table_{type_of_data}_*"))[-1]
results_table = pd.read_csv(fn)

# ------------------------------------------------------------
# Calculate coverage
coverage_table = []
if type_of_data == "one_brain": 
    true_dyads = pd.read_csv(Path("data") / "dyadList.csv")
    dyads = pseudodyads(true_dyads) 
    for i, row in dyads.iterrows(): 
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
                    classes = np.stack((target, partner), axis=1)
                    for class_combination in combinations_with_replacement(np.unique(classes), 2):
                        n = np.sum(classes[:,0] == class_combination[0] & (classes[:,1] == class_combination[1])) / len(target)
                        coverage_table.append(
                        [n, dyadID, dyadType, group, session, activity, str(class_combination[0]) + "_" + str(class_combination[1])]
                        )
else: 
    for id in sorted(set(results_table.ids)):
        if (len(str(id)) == 4): 
            dyadType = True
            if str(id).startswith("1"): 
                group = "same"
            else:
                group = "inter"
        else:
            dyadType = False
            group = "none"
        for session in sorted(set(results_table.sessions)):
            for activity in sorted(set(results_table.activities)):
                n_total = len(results_table[
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)])
                if n_total > 0:
                    for cluster in sorted(set(results_table.classes)):          
                        n = len(results_table[
                            (results_table.classes == cluster) & 
                            (results_table.activities == activity) & 
                            (results_table.sessions == session) & 
                            (results_table.ids == id)]) / n_total
                        coverage_table.append(
                        [n, id, dyadType, group, session, activity, cluster]
                        )
coverage_table = pd.DataFrame(coverage_table, 
             columns = ['n', 'id', 'dyadType', 'group', 'session', 'activity', 'cluster'])

# ------------------------------------------------------------
### Save results. 

coverage_table.to_csv((path / f"coverage_table_{type_of_data}.csv"), index=False)
    
