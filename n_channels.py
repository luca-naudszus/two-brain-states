# Investigating number of working channels per ID and session

import numpy as np
import pandas as pd

path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code/data/"
inpath = str(path + "/sourcedata/")

BESTchannels = pd.read_csv(str(path + "fNIRS_best_channels.csv"))
BESTchannels.dropna(axis = 0, how = 'any', inplace = True)
BESTchannels['session'] = BESTchannels['session'].astype(int)
BESTchannels['ID'] = BESTchannels['ID'].astype(int)

roi_counts = BESTchannels.groupby(['ID', 'session']).size().reset_index(name='n_rois')

roi_counts[roi_counts['n_rois'] == 8][['ID', 'session']]


