# Author: Luca A. Naudszus, Social Brain Sciences, ETH Zurich
# Date: 11 March 2025

# ------------------------------------------------------------
# Import packages and custom functions

from pathlib import Path
#---
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Set variables

path = 'results'
filename = 'results_table_two-blocks_2025-03-11.csv'

# ------------------------------------------------------------
# Load Data
results_table = pd.DataFrame(Path(path) / filename)

# ------------------------------------------------------------
# Calculate coverage
coverage_table = []
for id in sorted(set(results_table.ids)):
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
                    [n, id, session, activity, cluster]
                    )
coverage_table = pd.DataFrame(coverage_table, 
             columns = ['n', 'id', 'session', 'activity', 'cluster'])

# ------------------------------------------------------------
### Save results. 

coverage_table.to_csv(Path(path) / "coverage_table.csv", index=False)
    
