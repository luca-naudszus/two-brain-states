# Preparing data for clustering

#**Author:** Luca A. Naudszus
#**Date:** January 17, 2025
#**Affiliation:** Social Brain Sciences, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch

#TODO: structure this script and simplify it
from datetime import datetime
import mne
import numpy as np
import os
import pandas as pd
import re
import scipy as sp

# ------------------------------------------------------------
# Set variables
path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code/data/"

too_many_zeros = 100 # number of zeros in time series that is considered conspicuous
upsampling_freq = 5
verbosity = 40 # 40 = We will receive ERRORS and CRITICAL, but not WARNING, INFO, and DEBUG. 
which_freq_bands = 0 # Choose from: 0, 1, 2, 3. Frequency bands are below. 
freq_bands = [[0.015, 0.4], [0.1, 0.2], [0.03, 0.1], [0.02, 0.03]]

# ------------------------------------------------------------
# Define custom functions
#TODO: Let ChatGPT check these functions. 
def check_for_missing(ts):
    return np.isnan(ts).sum() > 0 or (ts == 0).sum() > too_many_zeros

def set_data_dict(dict, key, interpolation):
    dict[key] = {
            'interpolation': interpolation,
            'duration': duration_list,
            'true_duration_list': true_duration_list,
            'channels': channel_dict[key]}
    return dict

def get_partner_info(targetID, dyads):
    if dyads.pID1.isin([targetID]).any():
        partnerID = dyads.pID2[dyads.pID1 == targetID].iloc[0]
        dyadID = dyads.dyadID[dyads.pID1 == targetID].values
    elif dyads.pID2.isin([targetID]).any():
        partnerID = dyads.pID1[dyads.pID2 == targetID].iloc[0]
        dyadID = dyads.dyadID[dyads.pID2 == targetID].values
    else: 
        partnerID, dyadID = None, None
    return partnerID, dyadID

def interpolate_timeseries(source_ts, target_length): 
    ### Interpolate partner timeseries to length of target time series
    x_axis = np.arange(source_ts.shape[1])
    interpolated = np.zeros([source_ts.shape[0], target_length])
    for in_channel in range(source_ts.shape[0]):
        interpolated[in_channel] = sp.interpolate.CubicSpline(x_axis, source_ts[in_channel])(np.linspace(x_axis.min(), x_axis.max(), target_length))
    return interpolated

def compute_true_duration(dyadID, session_n, in_epoch, cutpoints, upsampling_freq):
    ### Get true start and end
    filtered_cutpoints = cutpoints[(cutpoints.Pair == dyadID) & (cutpoints.Session == session_n)]
    current_start, next_start = filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 1], filtered_cutpoints.Start[filtered_cutpoints.Task == in_epoch + 2]
                
    ### If this is known, find out task duration. 
    if not current_start.empty and not next_start.empty and not current_start.isna().iloc[0] and not next_start.isna().iloc[0]:
        current_start_time = datetime.combine(datetime(1, 1, 1), current_start.iloc[0])
        next_start_time = datetime.combine(datetime(1, 1, 1), next_start.iloc[0])
        true_duration = int((next_start_time - current_start_time).total_seconds())
        task_duration = filtered_cutpoints.Length[filtered_cutpoints.Task == in_epoch + 1]
        task_duration_secs = (datetime.combine(datetime.min, task_duration.iloc[0]) - datetime.min).total_seconds()
    ### If not, we need to assume that the interpolated duration is correct. 
    else:
        true_duration = np.nan
        task_duration_secs = 300  

    return true_duration, round(task_duration_secs * upsampling_freq)

def get_and_zscore(list, activity): 
    ts = sp.stats.zscore(list[activity], axis=1, ddof=1)
    return ts

# ------------------------------------------------------------
# Load data
inpath = str(path + "/preprocesseddata/")
dyads = pd.read_csv(str(path + "dyadList.csv")) # list of true dyads
cutpoints = pd.read_excel(str(path + "cutpoints_videos.xlsx")) # list of activity durations from video data
best_channels = pd.read_csv(str(path + 'fNIRS_chs_ROIproximal.csv'))

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
        nirs_path = str(inpath + file)
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

### To this end, we move outside the snirf structure and work only on the timeseries.
key_list = list(all_dict.keys())
data_dict, original_lengths, durations = {}, [], []
for key in key_list:
    # ------------------------------------------------------------
    # Get information on target and partner
    target = all_dict[key]
    targetID = int(key[4:7])
    session_n = int(key[-1])

    partnerID, dyadID = get_partner_info(targetID, dyads)
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
            if check_for_missing(target_ts): 
                error_log.append((f'sub-{targetID}_session-{session_n}', 'nans or too many zeros in data'))
                error_log.append((partner_key, 'nans or too many zeros in partner data'))
                valid_data = False
                break
            
            ### Load partner time series
            partner_ts = partner[in_epoch].get_data(copy = False, verbose=verbosity)[0]
            if check_for_missing(partner_ts): 
                error_log.append((f'sub-{targetID}_session-{session_n}', 'nans or too many zeros in partner data'))
                error_log.append((partner_key, 'nans or too many zeros in data'))
                valid_data = False
                break

            ### save original lengths for later comparison
            original_lengths.append(
                [targetID, partnerID, session_n, in_epoch, np.shape(target_ts)[1], np.shape(partner_ts)[1]])
            
            # ------------------------------------------------------------
            # Find out which duration is longer, this will be the duration at which we aim.
            
            if np.shape(target_ts)[1] > np.shape(partner_ts)[1]:
                ### Interpolate partner timeseries to length of target time series
                partner_interp = interpolate_timeseries(partner_ts, target_ts.shape[1])
                target_interp = np.copy(target_ts)
                duration_list.append(np.shape(target_interp)[1]/upsampling_freq)
            
            elif np.shape(partner_ts)[1] > np.shape(target_ts)[0]:
                ### Interpolate target timeseries to length of partner time series
                target_interp = interpolate_timeseries(target_ts, partner_ts.shape[1])
                partner_interp = np.copy(partner_ts)
                duration_list.append(np.shape(partner_interp)[1]/upsampling_freq)
            
            else:
                ### If both time series have the same length, there is no need to interpolate
                target_interp = np.copy(target_ts)
                partner_interp = np.copy(partner_ts)
                duration_list.append(np.shape(target_interp)[1] / upsampling_freq)

            assert np.shape(target_interp)[1] == np.shape(partner_interp)[1], f"Interpolation for {targetID} has not properly worked"
            
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
            true_duration_list.append(true_duration)
            durations.append([targetID, session_n, in_epoch, duration_list[in_epoch], true_duration])
            target_list.append(target_interp)
            partner_list.append(partner_interp)

        # ------------------------------------------------------------
        # remove partner from key_list because they have been interpolated
        key_list.remove(partner_key)

        # ------------------------------------------------------------
        # Update dictionaries
        if not valid_data:
            continue
        data_dict = set_data_dict(data_dict, key, target_list)
        data_dict = set_data_dict(data_dict, partner_key, partner_list)

    # ------------------------------------------------------------
    # Compare recorded duration with true duration to assess which time has actually passed (see above). 
    # This enables us to determine the end of the activity and dismiss the recording after that. 
    else: 
        target_list = []
        duration_list = []
        true_duration_list = []
        
        for in_epoch in range(0, len(target)):
            target_ts = target[in_epoch].get_data(copy = False, verbose=verbosity)[0]
            duration_list.append(np.shape(target_interp)[1] / upsampling_freq)
            
            ### We only have information on true duration for first three activities: 
            if in_epoch != len(target)-1: 
                true_duration, task_samples = compute_true_duration(dyadID, session_n, in_epoch, cutpoints, upsampling_freq)
                ### Dismiss the recording after the end of the activity.
                target_interp = target_interp[:,task_samples]
            else:
                true_duration = np.nan

            ### Save results
            true_duration_list.append(true_duration)
            durations.append([targetID, session_n, in_epoch, duration_list[in_epoch], true_duration])
            target_list.append(target_interp)

        # ------------------------------------------------------------
        # Update dictionaries
        data_dict = set_data_dict(data_dict, key, target_list)
        error_log.append((f'sub-{targetID}_session-{session_n}', 'partner data missing'))

# ------------------------------------------------------------
# Make data frame containing original lengths of target and partner time series
df_lengths = pd.DataFrame(original_lengths)
df_lengths.columns = ['target', 'partner', 'session', 'task', 'target_length', 'partner_length']
df_lengths['ratio'] = df_lengths.target_length / df_lengths.partner_length
df_lengths.ratio = np.where(df_lengths.ratio < 1, 1 / df_lengths.ratio,
                            df_lengths.ratio)

# ------------------------------------------------------------
# Make data frame containing the true and interpolated durations of target and partner time series
df_durations = pd.DataFrame(durations)
df_durations.columns = ['target', 'session', 'task', 'interpolated_duration', 'true_duration']
df_durations['ratio'] = df_durations.interpolated_duration / df_durations.true_duration
df_durations.ratio = np.where(df_durations.ratio < 1, 1 / df_durations.ratio,
                            df_durations.ratio)

# ------------------------------------------------------------
# Concatenate data in a meaningful way

ts_one_brain, ts_two_blocks, ts_four_blocks = [], [], []
ts_one_brain_session, ts_two_blocks_session, ts_four_blocks_session = [], [], []
doc_one_brain, doc_two_blocks, doc_four_blocks = [], [], []
doc_one_brain_session, doc_two_blocks_session, doc_four_blocks_session = [], [], []
channels_one_brain, channels_two_blocks, channels_four_blocks = [], [], []
channels_one_brain_session, channels_two_blocks_session, channels_four_blocks_session = [], [], []
key_list = set(data_dict.keys())
for i, row in dyads.iterrows():

    for session in range(6):
        target_key = f"sub-{row['pID1']}_session-{session + 1}"
        
        if target_key in key_list: 
            ### 
            target_list = data_dict[target_key]['interpolation']
            target_channels = ['target' + channel for channel in data_dict[target_key]['channels']]
            partner_key = f"sub-{row['pID2']}_session-{session + 1}"
            ts_target_temp, ts_partner_temp, ts_two_temp, ts_four_temp = [], [], [], []
            for activity in range(4):
                target_ts = get_and_zscore(target_list, activity)
                ### 1a target, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                ts_one_brain.append(target_ts)
                doc_one_brain.extend([[row['pID1'], session, activity]])
                channels_one_brain.append(target_channels)
                ts_target_temp.append(target_ts)
            
                if partner_key in key_list: 
                    
                    partner_channels = ['partner ' + channel for channel in data_dict[partner_key]['channels']]
                    partner_ts = get_and_zscore(data_dict[partner_key]['interpolation'], activity)

                    ### 1a partner, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                    ts_one_brain.append(partner_ts)
                    doc_one_brain.extend([[row['pID2'], session, activity]])
                    channels_one_brain.append(partner_channels)
                    ts_partner_temp.append(partner_ts)
                    
                    ### 2a, two blocks: target + partner, channel-wise z-scored
                    ts_two = np.concatenate((target_ts, partner_ts), axis = 0)         
                    ts_two_blocks.append(ts_two)
                    doc_two_blocks.extend([[row['dyadID'], session, activity]])
                    channels_two_blocks.append(target_channels + partner_channels)
                    ts_two_temp.append(ts_two)
                    
                    ### 3a, four blocks: target HbO, partner HbO, target HbR, partner HbR
                    ts_four = np.concatenate((target_ts[:int(len(target_channels)/2)], partner_ts[:int(len(partner_channels)/2)], target_ts[int(len(target_channels)/2):len(target_channels)], partner_ts[int(len(partner_channels)/2):len(partner_channels)]), axis=0)
                    ts_four_blocks.append(ts_four)
                    doc_four_blocks.extend([[row['dyadID'], session, activity]])
                    channels_four_blocks.append(np.concatenate((target_channels[:int(len(target_channels)/2)], 
                                                               partner_channels[:int(len(partner_channels)/2)], 
                                                               target_channels[int(len(target_channels)/2):len(target_channels)], 
                                                               partner_channels[int(len(partner_channels)/2):len(partner_channels)])))
                    ts_four_temp.append(ts_four)
            
            ### 1b target, one brain data as above, channel- and session-wise z-scored
            ## session-wise z-scoring
            n_samples = [ts.shape[1] for ts in ts_target_temp]
            ts_target_temp = sp.stats.zscore(np.concatenate(ts_target_temp, axis = 1), ddof=1)
            ts_target_split = np.split(ts_target_temp, np.cumsum(n_samples)[:-1], axis=1)

            ## append to list
            for activity in range(4):
                ts_one_brain_session.append(ts_target_split[activity])
                doc_one_brain_session.extend([[row['pID1'], session, activity]])
                channels_one_brain_session.append(target_channels)
            if partner_key in key_list: 
                ### session-wise z-scoring
                n_samples = [ts.shape[1] for ts in ts_partner_temp]
                ts_partner_temp = sp.stats.zscore(np.concatenate(ts_partner_temp, axis=1), ddof=1)
                ts_partner_split = np.split(ts_partner_temp, np.cumsum(n_samples)[:-1], axis=1)

                n_samples = [ts.shape[1] for ts in ts_two_temp]
                ts_two_temp = sp.stats.zscore(np.concatenate(ts_two_temp, axis=1), ddof=1)
                ts_two_split = np.split(ts_two_temp, np.cumsum(n_samples)[:-1], axis=1)

                n_samples = [ts.shape[1] for ts in ts_four_temp]
                ts_four_temp = sp.stats.zscore(np.concatenate(ts_four_temp, axis=1), ddof=1)
                ts_four_split = np.split(ts_four_temp, np.cumsum(n_samples)[:-1], axis=1)
                
                ## append to list
                for activity in range(4):
                    
                    ### 1b partner, one brain data as above, channel- and session-wise z-scored
                    ts_one_brain_session.append(ts_partner_split[activity])
                    doc_one_brain_session.extend([[row['pID2'], session, activity]])
                    channels_one_brain_session.append(partner_channels)
                    
                    ### 2b, two blocks as above, channel- and session-wise z-scored
                    ts_two_blocks_session.append(ts_two_split[activity])
                    doc_two_blocks_session.extend([[row['dyadID'], session, activity]])
                    channels_two_blocks_session.append(target_channels + partner_channels)
                    
                    ### 3b, four blocks as above, channel- and session-wise z-scored
                    ts_four_blocks_session.append(ts_four_split[activity])
                    doc_four_blocks_session.extend([[row['dyadID'], session, activity]])
                    channels_four_blocks_session.append(np.concatenate((target_channels[:int(len(target_channels)/2)], 
                                                               partner_channels[:int(len(partner_channels)/2)], 
                                                               target_channels[int(len(target_channels)/2):len(target_channels)], 
                                                               partner_channels[int(len(partner_channels)/2):len(partner_channels)])))

# ------------------------------------------------------------
# Save data

### Turn documentations into data frame
doc_one_brain = pd.DataFrame(doc_one_brain)
doc_two_blocks = pd.DataFrame(doc_two_blocks)
doc_four_blocks = pd.DataFrame(doc_four_blocks)
doc_one_brain_session = pd.DataFrame(doc_one_brain_session)
doc_two_blocks_session = pd.DataFrame(doc_two_blocks_session)
doc_four_blocks_session = pd.DataFrame(doc_four_blocks_session)

### Save documentations
doc_one_brain.to_csv(str(path + 'doc_one_brain.csv'))
doc_two_blocks.to_csv(str(path + 'doc_two_blocks.csv'))
doc_four_blocks.to_csv(str(path + 'doc_four_blocks.csv'))
doc_one_brain_session.to_csv(str(path + 'doc_one_brain_session.csv'))
doc_two_blocks_session.to_csv(str(path + 'doc_two_blocks_session.csv'))
doc_four_blocks_session.to_csv(str(path + 'doc_four_blocks_session.csv'))

### Save time series
np.savez(str(path + f'ts_one_brain_fb{which_freq_bands}'), *ts_one_brain)
np.savez(str(path + f'ts_two_blocks_fb{which_freq_bands}'), *ts_two_blocks)
np.savez(str(path + f'ts_four_blocks_fb{which_freq_bands}'), *ts_four_blocks)
np.savez(str(path + f'ts_one_brain_session_fb{which_freq_bands}'), *ts_one_brain_session)
np.savez(str(path + f'ts_two_blocks_session_fb{which_freq_bands}'), *ts_two_blocks_session)
np.savez(str(path + f'ts_four_blocks_session_fb{which_freq_bands}'), *ts_four_blocks_session)

### Save information on channels
np.savez(str(path + f'channels_one_brain_'), *channels_one_brain)
np.savez(str(path + f'channels_two_blocks'), *channels_two_blocks)
np.savez(str(path + f'channels_four_blocks'), *channels_four_blocks)
np.savez(str(path + f'channels_one_brain_session'), *channels_one_brain_session)
np.savez(str(path + f'channels_two_blocks_session'), *channels_two_blocks_session)
np.savez(str(path + f'channels_four_blocks_session'), *channels_four_blocks_session)