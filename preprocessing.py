# Identifying event states in two-person fNIRS data using Riemannian-geometry based k-Means

#**Author:** Luca A. Naudszus  
#**Date:** January 11, 2025
#**Affiliation:** Social Brain Sciences, ETH Zürich  
#**Email:** luca.naudszus@gess.ethz.ch  

## Preparations

import os
import re

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy as sp
import mne

path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code/data/"
inpath = str(path + "/sourcedata/")

### custom functions
def reshape(ts, upsampling_freq, window_length): 
    trim_size = ts.shape[1] % (upsampling_freq*window_length)
    ts_trimmed = ts[:, :-trim_size] if trim_size != 0 else ts
    ts_reshaped = ts_trimmed.reshape(-1, ts.shape[0], (upsampling_freq*window_length))
    return ts_reshaped

## Import data

### get list of true dyads
dyads = pd.read_csv(str(path + "dyadList.csv"))

### get list of best channels per ROI
BESTchannels = pd.read_csv(str(path + "best_channels.csv"))
BESTchannels.dropna(axis = 0, how = 'any', inplace = True)
BESTchannels['session'] = BESTchannels['session'].astype(int)
BESTchannels['ID'] = BESTchannels['ID'].astype(int)

### get list of activity durations (here extracted from video data)
cutpoints = pd.read_excel(str(path + "cutpoints_videos.xlsx"))

upsampling_freq = 100
window_length = 10 # virtual trial length in s

### load data and do first preprocessing steps
error_log = []
all_dict = {}
for file in os.listdir(inpath):
    if file.endswith(".snirf"):
        # for now, do not work with sessions with interrupted recording
        if re.match(r".*_[0-9]\.snirf$", file):
            error_log.append((file[:-8], 'interrupted recording'))
            continue
        
        # get path and info
        nirs_path = str(inpath + file)
        pID = int(file[4:7])
        session_n = int(file[-7:-6])
        #print("*** " + str(pID) + " " + str(session_n))
        new_name = file[:-6] + "_pre"

        # extract best channels
        best_chs = BESTchannels.loc[(BESTchannels['ID'] == pID) & 
                                    (BESTchannels['session'] == session_n)]
        # preprocess data
        raw = mne.io.read_raw_snirf(nirs_path)

        # convert to OD
        raw_od = mne.preprocessing.nirs.optical_density(raw)

        # apply motion correction
        raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

        # select channels
        chs850 = list(best_chs['channel'] + " 850")
        chs760 = list(best_chs['channel'] + " 760")
        keep_chs = chs850 + chs760
        if len(keep_chs) != len(set(keep_chs)): 
            error_log.append((file[:-6], 'duplicate best channels'))
            continue
        if len(keep_chs) == 0: 
            error_log.append((file[:-6], 'no information on best channels'))
            continue
        raw_od_good_chs = raw_od.pick(keep_chs)

        # convert to haemoglobin
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od_good_chs, ppf = 6.0)
        
        # check whether there are enough onsets recorded
        if len(raw_haemo.annotations) < 4:
            error_log.append((file[:-6], 'no onsets'))
            continue

        # define durations of epochs
        for in_activity in range(0, 3):
            raw_haemo.annotations.duration[
                in_activity] = round(raw_haemo.annotations.onset[
                in_activity + 1] - raw_haemo.annotations.onset[in_activity], 3)
            raw_haemo.annotations.duration[3] = round(raw_haemo[
            raw_haemo.ch_names[0]][1].max() - raw_haemo.annotations.onset[3], 3)
        # TO-DO: how do we handle too short durations? 
        # Especially: 103-1, less than 300 s for no visible reason

        # epoch data
        epoch_list = []
        for annot in raw_haemo.annotations:
            epoch = mne.Epochs(raw_haemo, 
                           mne.events_from_annotations(raw_haemo)[0][[0]],
                           tmin = 0,
                           tmax = annot['duration'],
                           baseline = None, 
                           verbose = 'WARNING')
            epoch_list.append(epoch)
        
        for epoch in epoch_list:
            epoch.load_data(verbose = 'WARNING')
            # upsample data
            epoch.resample(100, verbose = 'WARNING')
            
            # zero-phase 3rd order Butterworth filters to remove 0.03 - 0.1 Hz
            epoch.filter(0.02, 0.2, method='iir', iir_params=dict(order=3, ftype='butter'), verbose = 'WARNING')  # Keeps 0.02-0.2 Hz
            epoch.filter(0.03, 0.1, method='iir', iir_params=dict(order=3, ftype='butter', btype='bandstop'), verbose = 'WARNING')  # Removes 0.03-0.1 Hz
        
        # write into dictionary
        keyname = file[:-6]
        all_dict[keyname] = epoch_list

# align onsets
# To this end, we move outside the snirf structure and work only on the timeseries. 
key_list = list(all_dict.keys())
data_dict = {}
original_lengths = []
durations = []
for key in key_list:
    # load snirfs
    target = all_dict[key]
    targetID = int(key[4:7])
    session_n = int(key[-1])
    if dyads.pID1.isin([targetID]).any():
        partnerID = dyads.pID2[dyads.pID1 == targetID].iloc[0]
        dyadID = dyads.dyadID[dyads.pID1 == targetID].values
    elif dyads.pID2.isin([targetID]).any():
        partnerID = dyads.pID1[dyads.pID2 == targetID].iloc[0]
        dyadID = dyads.dyadID[dyads.pID2 == targetID].values
    partner_key = f'sub-{partnerID}_session-{session_n}'
    
    if partner_key not in all_dict:
        error_log.append((f'sub-{targetID}_session-{session_n}', 'partner data missing'))
        continue
    
    partner = all_dict[partner_key]
    
    target_list = []
    partner_list = []
    duration_list = []
    true_duration_list = []
    
    # process each epoch
    for in_epoch in range(0, len(target)):
        target_interp = []
        partner_interp = []
        target_epoch = target[in_epoch]
        target_ts = target_epoch.get_data(copy = False)[0]
        partner_epoch = partner[in_epoch]
        partner_ts = partner_epoch.get_data(copy = False)[0]
        original_lengths.append(
            [targetID, partnerID, session_n, in_epoch, np.shape(target_ts)[1], np.shape(partner_ts)[1]])
        # Find out which duration is longer, this will be the duration at which we aim. 

        if np.shape(target_ts)[1] > np.shape(partner_ts)[1]:
            # interpolate partner timeseries to length of target time series
            x_axis = np.arange(np.shape(partner_ts)[1])
            partner_interp = np.copy(target_ts)
            for in_channel in range(0, 8): 
                partner_interp[in_channel] = sp.interpolate.CubicSpline(x_axis, partner_ts[in_channel])(np.linspace(x_axis.min(), x_axis.max(), np.shape(target_ts)[1]))
            # keep target timeseries
            target_interp = np.copy(target_ts)
            # write duration into list
            duration_list.append(np.shape(target_interp)[1]/upsampling_freq)
        elif np.shape(partner_ts)[1] > np.shape(target_ts)[0]:
            x_axis = np.arange(np.shape(target_ts)[1])
            target_interp = np.copy(partner_ts)
            # interpolate target timeseries to length of partner time series
            for in_channel in range(0, 8):
                target_interp[in_channel] = sp.interpolate.CubicSpline(x_axis, target_ts[in_channel])(np.linspace(x_axis.min(), x_axis.max(), np.shape(partner_ts)[1]))
            # keep partner timeseries
            partner_interp = np.copy(partner_ts)
            # write duration into list
            duration_list.append(np.shape(partner_interp)[1]/upsampling_freq)
        else: 
            target_interp = np.copy(target_ts)
            partner_interp = np.copy(partner_ts)
            duration_list.append(np.shape(target_interp)[1] / upsampling_freq)
            
        assert np.shape(target_interp)[1] == np.shape(partner_interp)[1], "Interpolation has not properly worked"

        # Compare the interpolated duration with the true duration
        if in_epoch != len(target)-1:
            filtered_cutpoints = cutpoints[(cutpoints.Pair.values == dyadID) & (cutpoints.Session.values == session_n)]
            current_start = filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 1]
            next_start = filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 2]
            if not current_start.empty and not next_start.empty and not current_start.isna().iloc[0] and not next_start.isna().iloc[0]:
                current_start_time = datetime.combine(datetime(1, 1, 1), current_start.iloc[0])
                next_start_time = datetime.combine(datetime(1, 1, 1), next_start.iloc[0])
                true_duration = int((next_start_time - current_start_time).total_seconds())
                task_duration = filtered_cutpoints.Length[filtered_cutpoints.Task == in_epoch + 1]
                task_duration_secs = (datetime.combine(datetime.min, task_duration.iloc[0]) - datetime.min).total_seconds() 
            else: 
                true_duration = np.nan
                task_duration_secs = 300
            
            # Now we can dismiss the recording after the end of the activity. 
            target_interp = target_interp[:,:round(task_duration_secs*upsampling_freq)]
            partner_interp = partner_interp[:,:round(task_duration_secs*upsampling_freq)]
        else: 
            true_duration = np.nan
        true_duration_list.append(true_duration)
        durations.append([targetID, session_n, in_epoch, duration_list[in_epoch], true_duration])      
        
        target_list.append(target_interp)
        partner_list.append(partner_interp)
        
    # update dictionaries
    data_dict[key] = {
        'interpolation': target_list,
        'duration': duration_list,
        'true_duration': true_duration_list}
    data_dict[partner_key] = {
        'interpolation': partner_list,
        'duration': duration_list,
        'true_duration_list': true_duration_list}
    # remove partner from key_list because they have been interpolated
    key_list.remove(partner_key)

df_lengths = pd.DataFrame(original_lengths)
df_lengths.columns = ['target', 'partner', 'session', 'task', 'target_length', 'partner_length']
df_lengths['ratio'] = df_lengths.target_length / df_lengths.partner_length
df_lengths.ratio = np.where(df_lengths.ratio < 1, 1 / df_lengths.ratio, 
                            df_lengths.ratio)

df_durations = pd.DataFrame(durations)
df_durations.columns = ['target', 'session', 'task', 'interpolated_duration', 'true_duration']
df_durations['ratio'] = df_durations.interpolated_duration / df_durations.true_duration
df_durations.ratio = np.where(df_durations.ratio < 1, 1 / df_durations.ratio, 
                            df_durations.ratio)

ts_one_brain, ts_two_blocks, ts_four_blocks = [], [], []
doc_one_brain, doc_two_blocks, doc_four_blocks = [], [], []
key_list = set(all_dict.keys())

for i, row in dyads.iterrows():
    for session in range(6):
        target_key = f"sub-{row['pID1']}_session-{session + 1}"
        partner_key = f"sub-{row['pID2']}_session-{session + 1}"

        if target_key in key_list and partner_key in key_list:
            target_list, partner_list = data_dict[target_key]['interpolation'], data_dict[partner_key]['interpolation']

            for activity in range(4):
                target_ts, partner_ts = target_list[activity], partner_list[activity]

                # first, one-brain data
                for ts, pID in zip([target_ts, partner_ts], [row['pID1'], row['pID2']]):
                    ts_reshaped = reshape(ts, upsampling_freq, window_length)
                    ts_one_brain.append(ts_reshaped)
                    doc_one_brain.extend([[pID, session, activity]] * ts_reshaped.shape[0])

                # second, two blocks: target + partner
                ts_two_reshaped = reshape(np.concatenate((target_ts, partner_ts), axis=0), upsampling_freq, window_length)
                ts_two_blocks.append(ts_two_reshaped)
                doc_two_blocks.extend([[row['dyadID'], session, activity]] * ts_two_reshaped.shape[0])

                # third, four blocks: target HbO, partner HbO, target HbR, partner HbR
                ts_four_reshaped = reshape(np.concatenate((target_ts[:4], partner_ts[:4], target_ts[4:8], partner_ts[4:8]), axis=0), upsampling_freq, window_length)
                ts_four_blocks.append(ts_four_reshaped)
                doc_four_blocks.extend([[row['dyadID'], session, activity]] * ts_four_reshaped.shape[0])

matrix_one_brain = np.concatenate(ts_one_brain, axis = 0)
print(
    f"Data for single brain preprocessed: {matrix_one_brain.shape[0]} trials, {matrix_one_brain.shape[1]} channels, "
    f"{matrix_one_brain.shape[2]} time points"
)
matrix_two_blocks = np.concatenate(ts_two_blocks, axis = 0)
print(
    f"Data for two blocks preprocessed: {matrix_two_blocks.shape[0]} trials, {matrix_two_blocks.shape[1]} channels, "
    f"{matrix_two_blocks.shape[2]} time points"
)
matrix_four_blocks = np.concatenate(ts_four_blocks, axis = 0)
print(
    f"Data for four blocks preprocessed: {matrix_four_blocks.shape[0]} trials, {matrix_four_blocks.shape[1]} channels, "
    f"{matrix_four_blocks.shape[2]} time points"
)

doc_one_brain = pd.DataFrame(doc_one_brain)
doc_two_blocks = pd.DataFrame(doc_two_blocks)
doc_four_blocks = pd.DataFrame(doc_four_blocks)

print(
    f"Data preprocessed: {matrix_four_blocks.shape[0]} trials, {matrix_four_blocks.shape[1]} channels, "
    f"{matrix_four_blocks.shape[2]} time points"
)

# save 
doc_one_brain.to_csv(str(path + 'doc_one_brain.csv'))
np.save(str(path + 'matrix_one_brain'), matrix_one_brain)
doc_two_blocks.to_csv(str(path + 'doc_two_blocks.csv'))
np.save(str(path + 'matrix_two_blocks'), matrix_two_blocks)
doc_four_blocks.to_csv(str(path + 'doc_four_blocks.csv'))
np.save(str(path + 'matrix_four_blocks'), matrix_four_blocks)