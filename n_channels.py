# Investigating number of working channels per ID and session

#**Author:** Luca A. Naudszus
#**Date:** 29 January 2025
#**Affiliation:** Social Brain Sciences Lab, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch


# ------------------------------------------------------------
# Import packages
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Set paths
path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/analysis/data/time-series-features/clustering/fNIRS_prepared/additional-information"

# ------------------------------------------------------------
# Read best channel data
BESTchannels = pd.read_csv(str(path + "fNIRS_best_channels.csv"))
BESTchannels.dropna(axis = 0, how = 'any', inplace = True)
BESTchannels['session'] = BESTchannels['session'].astype(int)
BESTchannels['ID'] = BESTchannels['ID'].astype(int)

# ------------------------------------------------------------
# Count per ROI
roi_counts = BESTchannels.groupby(['ID', 'session']).size().reset_index(name='n_rois')
roi_counts[roi_counts['n_rois'] == 8][['ID', 'session']]


