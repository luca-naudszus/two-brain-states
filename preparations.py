# Preparing data for clustering

#**Author:** Luca A. Naudszus
#**Date:** 6 March 2025
#**Affiliation:** Social Brain Sciences Lab, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch

# ------------------------------------------------------------
# Import packages
from datetime import datetime, timedelta
from pathlib import Path
import os
import re
#---
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy as sp
#---
import mne
from riemannianKMeans import pseudodyads

# ------------------------------------------------------------
# Set variables
path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code/data"
#path = "./data"

pseudo_dyads = True # Create pseudo dyads
session_wise = False

too_many_zeros = 100 # number of zeros in time series that is considered conspicuous
upsampling_freq = 5
verbosity = 40 # 40 = We will receive ERRORS and CRITICAL, but not WARNING, INFO, and DEBUG. 
which_freq_bands = 0 # Choose from: 0, 1, 2, 3. Frequency bands are below. 
freq_bands = [[0.015, 0.4], [0.1, 0.2], [0.03, 0.1], [0.02, 0.03]]

# ------------------------------------------------------------
# Define custom functions
def check_for_missing(ts, too_many_zeros=too_many_zeros):
    return np.any(np.isnan(ts)) or (ts == 0).sum() > too_many_zeros

def set_data_dict(data_dict, key, interpolation, duration_list, true_duration_list, channel_dict):
    data_dict[key] = {
            'interpolation': interpolation,
            'duration': duration_list,
            'true_duration_list': true_duration_list,
            'channels': channel_dict[key]}
    return data_dict

def get_partner_info(targetID, dyads):
    row = dyads.query("pID1 == @targetID or pID2 == @targetID")
    if not row.empty:
        partnerID = row.pID2.iloc[0] if row.pID1.iloc[0] == targetID else row.pID1.iloc[0]
        dyadID = row.dyadID.tolist()
        return partnerID, dyadID
    return None, None

def interpolate_timeseries(source_ts, target_length): 
    ### Interpolate partner timeseries to length of target time series
    x_axis = np.arange(source_ts.shape[1])
    interpolated = np.empty([source_ts.shape[0], target_length])
    for in_channel in range(source_ts.shape[0]):
        interpolated[in_channel] = sp.interpolate.CubicSpline(x_axis, source_ts[in_channel])(np.linspace(0, source_ts.shape[1] - 1, target_length))
    return interpolated

def compute_true_duration(dyadID, session_n, in_epoch, cutpoints, upsampling_freq):
    ### Get true start and end
    filtered_cutpoints = cutpoints.query("Pair == @dyadID and Session == @session_n")
    current_start, next_start = filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 1], \
                                filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 2]
                
    ### If this is known, find out task duration. 
    if not current_start.empty and not next_start.empty and not current_start.isna().iloc[0] and not next_start.isna().iloc[0]:
        current_start_time = datetime.combine(datetime.min, current_start.iloc[0])
        next_start_time = datetime.combine(datetime.min, next_start.iloc[0])
        true_duration = int((next_start_time - current_start_time).total_seconds())
        task_duration = filtered_cutpoints.Length[filtered_cutpoints.Task == in_epoch + 1].iloc[0]
        task_duration_secs = timedelta(hours=task_duration.hour, minutes=task_duration.minute, seconds=task_duration.second).total_seconds()
    
    ### If not, we need to assume that the interpolated duration is correct. 
    else:
        true_duration, task_duration_secs = np.nan, 300  

    return true_duration, round(task_duration_secs * upsampling_freq)

def get_and_zscore(list, activity): 
    ts = sp.stats.zscore(list[activity], axis=1, ddof=1)
    return ts

def zscore_and_split(ts_list):
    n_samples = [ts.shape[1] for ts in ts_list]
    return np.split(sp.stats.zscore(np.concatenate(ts_list, axis=1), ddof=1),
                    np.cumsum(n_samples)[:-1], axis=1)

def add_durations(targetID, session_n, in_epoch, duration_epoch, true_duration):
    true_duration_list.append(true_duration)
    durations.append([targetID, session_n, in_epoch, duration_epoch, true_duration])

# ------------------------------------------------------------
# Load data
inpath = Path(path, "preprocesseddata")
true_dyads = pd.read_csv(Path(path, "dyadList.csv")) # list of true dyads
cutpoints = pd.read_excel(Path(path, "cutpoints_videos.xlsx")) # list of activity durations from video data
best_channels = pd.read_csv(Path(path, "fNIRS_chs_ROIproximal.csv"))

expected_rois = {'l_ifg', 'l_tpj', 'r_ifg', 'r_tpj'}

### Load single fif files
all_dict, roi_dict, channel_dict, error_log = {}, {}, {}, []
for file in os.listdir(inpath):
    if file.endswith(".fif"):
        
        # Test whether the recording has been broken
        #TODO: deal with broken recordings, n = 6,
        # this would give six more individual session data and four more dyad session data
        if re.match(r"^.+_.{1}_pre\.fif$", file):
            error_log.append((file[:-8], 'recording is broken'))
            continue

        # Get path and info
        nirs_path = Path(inpath, file)
        pID = int(file[4:7])
        session_n = int(file[16])

        # Read data
        data = mne.io.read_raw_fif(nirs_path, verbose=verbosity)

        # Check channels
        chs = data.info['ch_names']
        assert len(chs) == len(set(chs)), f"Duplicate channels for {file[:-8]}"
        assert len(chs) != 0, f"No channels for {file[:-8]}"
        best_chs = best_channels[(best_channels.ID == pID) & (best_channels.session == session_n)]
        # get ROIs (for some reason, channels in best_chs and data are different)
        #base_chs = pd.Series(chs).str.replace(r' (hbo|hbr)$', '', regex=True)
        #channel_to_ROI = dict(zip(best_chs['channel'], best_chs['ROI']))
        #ROI_array = pd.Series(base_chs).map(channel_to_ROI)

        # Check whether there are enough onsets recorded
        if len(data.annotations) < 4:
            # we need to exclude these data
            error_log.append((file[:-8], 'no onsets'))
            continue

        # Define durations of epochs
        events, event_dict = mne.events_from_annotations(data, verbose=verbosity)
        for in_activity in range(3):
            data.annotations.duration[
                in_activity] = (events[in_activity + 1][0] - events[in_activity][0] - 1)/5
        data.annotations.duration[3] = (data[data.ch_names[0]][0].shape[1] - events[3][0] - 1)/5

        # Epoch data
        epoch_list = []
        i = 0
        
        for annot in data.annotations:
            epoch = mne.Epochs(data,
                           events[[i]],
                           tmin = 0,
                           tmax = annot['duration'],
                           baseline = None,
                           preload = which_freq_bands != 0,
                           verbose=verbosity
                           )
            if which_freq_bands != 0:
                epoch.filter(l_freq = freq_bands[which_freq_bands][0], 
                         h_freq = freq_bands[which_freq_bands][1], 
                         verbose=verbosity)
            assert len(epoch.drop_log[0]) == 0, 'Dropped all epochs here'
            epoch_list.append(epoch)
            i += 1

        # Write into dictionary
        keyname = file[:17]
        all_dict[keyname] = epoch_list
        channel_dict[keyname] = chs

# ------------------------------------------------------------
# Align onsets

### To this end, we move outside the fif structure and work only on the timeseries.
key_list = set(all_dict.keys())
data_dict, original_lengths, durations, preprocessed_keys = {}, [], [], set()
for key in key_list:
    if key in preprocessed_keys: 
        continue
    # ------------------------------------------------------------
    # Get information on target and partner
    target = all_dict[key]
    targetID = int(key[4:7])
    session_n = int(key[-1])

    partnerID, dyadID = get_partner_info(targetID, true_dyads)
    partner_key = f'sub-{partnerID}_session-{session_n}'

    # ------------------------------------------------------------
    # We can only align if both target and partner data exist. 
    if partner_key in all_dict:
        valid_data = True
        partner = all_dict[partner_key]
        target_list, partner_list, duration_list, true_duration_list = [], [], [], []

        # ------------------------------------------------------------
        # Process each epoch
        for in_epoch in range(0, len(target)):
            target_interp, partner_interp = [], []
            
            ### Load target time series
            target_ts = target[in_epoch].get_data(copy = False, verbose=verbosity)[0]
            
            ### Load partner time series
            partner_ts = partner[in_epoch].get_data(copy = False, verbose=verbosity)[0]
            
            ### Check if data is missing
            if check_for_missing(partner_ts) or check_for_missing(target_ts): 
                error_log.extend([
                    (f'sub-{targetID}_session-{session_n}', 'nans or too many zeros in target or partner data'),
                    (partner_key, 'nans or too many zeros in target or partner data')
                    ])
                valid_data = False
                break

            ### save original lengths for later comparison
            original_lengths.append(
                [targetID, partnerID, session_n, in_epoch, np.shape(target_ts)[1], np.shape(partner_ts)[1]])
            
            # ------------------------------------------------------------
            # Interpolate timeseries
            
            ### Find out which duration is longer, this will be the duration at which we aim.
            len_target, len_partner = target_ts.shape[1], partner_ts.shape[1]
            len_interp = max(len_target, len_partner)

            ### Interpolate timeseries with shorter duration (unless both have equal length)
            target_interp = target_ts if len_target == len_interp else interpolate_timeseries(target_ts, len_interp)
            partner_interp = partner_ts if len_partner == len_interp else interpolate_timeseries(partner_ts, len_interp)

            duration_list.append(len_interp / upsampling_freq)

            assert target_interp.shape[1] == partner_interp.shape[1], f"Interpolation for {targetID} has not properly worked"
            
            # ------------------------------------------------------------
            # Compare interpolated duration with true duration to assess which time has actually passed
            # This enables us to determine the end of the activity and dismiss the recording after that. 

            ### We only have information on true duration for first three activities: 
            if in_epoch != len(target)-1:
                
                true_duration, task_samples = compute_true_duration(dyadID, session_n, in_epoch, cutpoints, upsampling_freq)
                ### Dismiss the recording after the end of the activity.
                target_interp = target_interp[:,:task_samples]
                partner_interp = partner_interp[:,:task_samples]
            else:
                true_duration = np.nan

            ### Save results
            add_durations(targetID, session_n, in_epoch, duration_list[in_epoch], true_duration)
            target_list.append(target_interp)
            partner_list.append(partner_interp)

        # ------------------------------------------------------------
        # remove partner from key_list because they have been interpolated
        preprocessed_keys.add(partner_key)

        # ------------------------------------------------------------
        # Update dictionaries
        if not valid_data:
            continue
        data_dict = set_data_dict(data_dict, key, target_list, duration_list, true_duration_list, channel_dict)
        data_dict = set_data_dict(data_dict, partner_key, partner_list, duration_list, true_duration_list, channel_dict)

    # ------------------------------------------------------------
    # Compare recorded duration with true duration to assess which time has actually passed (see above). 
    # This enables us to determine the end of the activity and dismiss the recording after that. 
    else: 
        target_list = []
        duration_list = []
        true_duration_list = []
        
        for in_epoch in range(0, len(target)):
            target_interp = target[in_epoch].get_data(copy = False, verbose=verbosity)[0]
            duration_list.append(target_interp.shape[1] / upsampling_freq)
            
            ### We only have information on true duration for first three activities: 
            if in_epoch != len(target)-1: 
                true_duration, task_samples = compute_true_duration(dyadID, session_n, in_epoch, cutpoints, upsampling_freq)
                ### Dismiss the recording after the end of the activity.
                target_interp = target_interp[:,:task_samples]
            else:
                true_duration = np.nan

            ### Save results
            add_durations(targetID, session_n, in_epoch, duration_list[in_epoch], true_duration)
            target_list.append(target_interp)

        # ------------------------------------------------------------
        # Update dictionaries
        data_dict = set_data_dict(data_dict, key, target_list, duration_list, true_duration_list, channel_dict)
        error_log.append((f'sub-{targetID}_session-{session_n}', 'partner data missing'))

# ------------------------------------------------------------
# Make data frame containing the original lengths, and true and interpolated durations of target and partner time series
df_lengths, df_durations = pd.DataFrame(original_lengths), pd.DataFrame(durations)
df_durations.columns = ['target', 'session', 'task', 'interpolated_duration', 'true_duration']
df_lengths.columns = ['target', 'partner', 'session', 'task', 'target_length', 'partner_length']
for df, num, denom in [(df_lengths, 'target_length', 'partner_length'),
                        (df_durations, 'interpolated_duration', 'true_duration')]:
    df['ratio'] = df[num] / df[denom]
    df['ratio'] = np.maximum(df['ratio'], 1 / df['ratio'])

# ------------------------------------------------------------
# Concatenate data in a meaningful way

ts = defaultdict(lambda: defaultdict(list))
doc, channels = defaultdict(list), defaultdict(list)
key_list = set(data_dict.keys())
dyads = pseudodyads(true_dyads) if pseudo_dyads else true_dyads
for i, row in dyads.iterrows():
    is_real = row['dyadType'] if pseudo_dyads else True
    dyadID = row['dyadID']
    group = row['group'] if pseudo_dyads else ("same" if dyadID < 2000 else "inter")

    for session in range(6):
        target_key, partner_key = (f"sub-{row[pID]}_session-{session + 1}" for pID in ['pID1', 'pID2'])
        if target_key in key_list: 
            ### 
            target_list = data_dict[target_key]['interpolation']
            target_channels = ['target ' + channel for channel in data_dict[target_key]['channels']]
            ts_target_temp, ts_partner_temp, ts_two_temp, ts_four_temp = [], [], [], []

            for activity in range(4):
                target_ts = get_and_zscore(target_list, activity).astype(np.float32)
                
                ### 1a target, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                if is_real:
                    ts["one_brain"]["channel-wise"].append(target_ts)
                    doc["one_brain"].extend([[row['pID1'], session, activity]])
                    channels["one_brain"].append(target_channels)
                ts_target_temp.append(target_ts)

                if partner_key in key_list: 
                    
                    partner_channels = ['partner ' + channel for channel in data_dict[partner_key]['channels']]
                    partner_ts = get_and_zscore(data_dict[partner_key]['interpolation'], activity).astype(np.float32)

                    ### 1a partner, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                    if is_real:
                        ts["one_brain"]["channel-wise"].append(partner_ts)
                        doc["one_brain"].extend([[row['pID2'], session, activity]])
                        channels["one_brain"].append(partner_channels)
                    ts_partner_temp.append(partner_ts)


                    if not is_real: 
                        n_samples = min(target_ts.shape[1], partner_ts.shape[1])
                        target_ts, partner_ts = target_ts[:,:n_samples], partner_ts[:,:n_samples]
                    
                    ### 2a, two blocks: target + partner, channel-wise z-scored
                    ts_two = np.concatenate((target_ts, partner_ts), axis=0).astype(np.float32)     
                    ts["two_blocks"]["channel-wise"].append(ts_two)
                    doc["two_blocks"].extend([[dyadID, is_real, group, session, activity]])
                    channels["two_blocks"].append(target_channels + partner_channels)
                    ts_two_temp.append(ts_two)

                    ### 3a, four blocks: target HbO, partner HbO, target HbR, partner HbR
                    lentchs, lenpchs = int(len(target_channels)), int(len(partner_channels))
                    ts_four = np.concatenate((target_ts[:int(lentchs/2)], 
                                              partner_ts[:int(lenpchs/2)], 
                                              target_ts[int(lentchs/2):lentchs], 
                                              partner_ts[int(lenpchs/2):lenpchs]), axis=0).astype(np.float32)
                    ts["four_blocks"]["channel-wise"].append(ts_four)
                    doc["four_blocks"].extend([[dyadID, is_real, group, session, activity]])
                    channels["four_blocks"].append(np.concatenate((target_channels[:int(lentchs/2)], 
                                                               partner_channels[:int(lenpchs/2)], 
                                                               target_channels[int(lentchs/2):lentchs], 
                                                               partner_channels[int(lenpchs/2):lenpchs]), axis=0))
                    ts_four_temp.append(ts_four)
            
            ### 1b target, one brain data as above, channel- and session-wise z-scored
            ## session-wise z-scoring
            if session_wise: 
                ts_target_split = zscore_and_split(ts_target_temp)

                ## append to list
                if is_real:
                    for activity in range(4):
                        ts["one_brain"]["session-wise"].append(ts_target_split[activity])

                if partner_key in key_list: 
                    ### session-wise z-scoring
                    ts_partner_split, ts_two_split, ts_four_split = map(zscore_and_split, 
                                                    [ts_partner_temp, ts_two_temp, ts_four_temp])
                
                    ## append to list
                    for activity in range(4):
                    
                        ### 1b partner, one brain data as above, channel- and session-wise z-scored
                        if is_real: 
                            ts["one_brain"]["session-wise"].append(ts_partner_split[activity].astype(np.float32))
                    
                        ### 2b, two blocks as above, channel- and session-wise z-scored
                        ts["two_blocks"]["session-wise"].append(ts_two_split[activity].astype(np.float32))        
                        ### 3b, four blocks as above, channel- and session-wise z-scored
                        ts["four_blocks"]["session-wise"].append(ts_four_split[activity].astype(np.float32))

# ------------------------------------------------------------
# Save data
pseudo = "true" if pseudo_dyads else "false"
for key in list(ts.keys()):
    np.savez(Path(path, f"ts_{key}_fb-{which_freq_bands}_pseudo-{pseudo}"), *ts[key]['channel-wise'])
    np.savez(Path(path, f"ts_{key}_session_fb-{which_freq_bands}_pseudo-{pseudo}"), *ts[key]['session-wise'])
    pd.DataFrame(doc[key]).to_csv(Path(path, f"doc_{key}_pseudo-{pseudo}.csv"))
    np.savez(Path(path, f"channels_{key}_pseudo-{pseudo}"), *channels[key])