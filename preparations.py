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

path = "/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code/data/"
inpath = str(path + "/preprocesseddata/")


### get list of true dyads
dyads = pd.read_csv(str(path + "dyadList.csv"))

### get list of activity durations (here extracted from video data)
cutpoints = pd.read_excel(str(path + "cutpoints_videos.xlsx"))

### get channels csv
best_channels = pd.read_csv(str(path + 'fNIRS_chs_ROIproximal.csv'))
expected_rois = {'l_ifg', 'l_tpj', 'r_ifg', 'r_tpj'}

too_many_zeros = 100
current_freq = 5
verbosity = 40 #ERRORS and CRITICAL, but not WARNING, INFO, DEBUG

all_dict = {}
roi_dict = {}
error_log = []
for file in os.listdir(inpath):
    if file.endswith(".fif"):

        # get path and info
        nirs_path = str(inpath + file)
        pID = int(file[4:7])
        session_n = int(file[-9:-8])

        # test whether the recording has been broken
        #TODO: deal with broken recordings, n = 6,
        # this would give six more individual session data and four more dyad session data
        if re.match(r"^.+_.{1}_pre\.fif$", file):
            error_log.append((file[:-8], 'recording is broken'))
            continue
        
        # read data
        data = mne.io.read_raw_fif(nirs_path, verbose=verbosity)

        # check channels
        chs = data.info['ch_names']
        assert len(chs) == len(set(chs)), f"Duplicate channels for {file[:-8]}"
        assert len(chs) != 0, f"No channels for {file[:-8]}"
        best_chs = best_channels[(best_channels.ID == pID) & (best_channels.session == session_n)]
        # get ROIs (for some reason, channels in best_chs and data are different)
        #base_chs = pd.Series(chs).str.replace(r' (hbo|hbr)$', '', regex=True)
        #channel_to_ROI = dict(zip(best_chs['channel'], best_chs['ROI']))
        #ROI_array = pd.Series(base_chs).map(channel_to_ROI)
        ROI_list = list(set(best_chs.ROI))

        # check whether there are enough onsets recorded
        if len(data.annotations) < 4:
            # we need to exclude these data
            error_log.append((file[:-8], 'no onsets'))
            continue

        # define durations of epochs
        events, event_dict = mne.events_from_annotations(data, verbose=verbosity)
        for in_activity in range(3):
            data.annotations.duration[
                in_activity] = (events[in_activity + 1][0] - events[in_activity][0] - 1)/5
        data.annotations.duration[3] = (data[data.ch_names[0]][0].shape[1] - events[3][0] - 1)/5

        # epoch data
        epoch_list = []
        i = 0
        
        for annot in data.annotations:
            epoch = mne.Epochs(data,
                           events[[i]],
                           tmin = 0,
                           tmax = annot['duration'],
                           baseline = None,
                           verbose=verbosity
                           )
            assert len(epoch.drop_log[0]) == 0, 'Dropped all epochs here'
            epoch_list.append(epoch)
            i += 1

        # write into dictionary
        keyname = file[:-8]
        all_dict[keyname] = epoch_list
        roi_dict[keyname] = ROI_list + ROI_list

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

    if partner_key in all_dict:
        valid_data = True
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
            target_ts = target_epoch.get_data(copy = False, verbose=verbosity)[0]
            if np.isnan(target_ts).sum() > 0 or (target_ts == 0).sum() > too_many_zeros: 
                error_log.append((f'sub-{targetID}_session-{session_n}', 'nans or too many zeros in data'))
                error_log.append((partner_key, 'nans or too many zeros in partner data'))
                valid_data = False
                break
            partner_epoch = partner[in_epoch]
            partner_ts = partner_epoch.get_data(copy = False, verbose=verbosity)[0]
            if np.isnan(partner_ts).sum() > 0 or (partner_ts == 0).sum() > too_many_zeros: 
                error_log.append((f'sub-{targetID}_session-{session_n}', 'nans or too many zeros in partner data'))
                error_log.append((partner_key, 'nans or too many zeros in data'))
                valid_data = False
                break
            original_lengths.append(
                [targetID, partnerID, session_n, in_epoch, np.shape(target_ts)[1], np.shape(partner_ts)[1]])
            # Find out which duration is longer, this will be the duration at which we aim.
            if np.shape(target_ts)[1] > np.shape(partner_ts)[1]:
                # interpolate partner timeseries to length of target time series
                x_axis = np.arange(np.shape(partner_ts)[1])
                partner_interp = np.zeros([partner_ts.shape[0], target_ts.shape[1]])
                for in_channel in range(0, len(partner_ts)):
                    partner_interp[in_channel] = sp.interpolate.CubicSpline(x_axis, partner_ts[in_channel])(np.linspace(x_axis.min(), x_axis.max(), np.shape(target_ts)[1]))
                # keep target timeseries
                target_interp = np.copy(target_ts)
                # write duration into list
                duration_list.append(np.shape(target_interp)[1]/current_freq)
            elif np.shape(partner_ts)[1] > np.shape(target_ts)[0]:
                x_axis = np.arange(np.shape(target_ts)[1])
                target_interp = np.zeros([target_ts.shape[0], partner_ts.shape[1]])
                # interpolate target timeseries to length of partner time series
                for in_channel in range(0, len(target_ts)):
                    target_interp[in_channel] = sp.interpolate.CubicSpline(x_axis, target_ts[in_channel])(np.linspace(x_axis.min(), x_axis.max(), np.shape(partner_ts)[1]))
                # keep partner timeseries
                partner_interp = np.copy(partner_ts)
                # write duration into list
                duration_list.append(np.shape(partner_interp)[1]/current_freq)
            else:
                target_interp = np.copy(target_ts)
                partner_interp = np.copy(partner_ts)
                duration_list.append(np.shape(target_interp)[1] / current_freq)

            assert np.shape(target_interp)[1] == np.shape(partner_interp)[1], f"Interpolation for {targetID} has not properly worked"

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
                target_interp = target_interp[:,:round(task_duration_secs*current_freq)]
                partner_interp = partner_interp[:,:round(task_duration_secs*current_freq)]
            else:
                true_duration = np.nan
            true_duration_list.append(true_duration)
            durations.append([targetID, session_n, in_epoch, duration_list[in_epoch], true_duration])

            target_list.append(target_interp)
            partner_list.append(partner_interp)

        # remove partner from key_list because they have been interpolated
        key_list.remove(partner_key)

        if not valid_data:
            continue
        # update dictionaries
        data_dict[key] = {
            'interpolation': target_list,
            'duration': duration_list,
            'true_duration': true_duration_list,
            'rois': roi_dict[key]}
        data_dict[partner_key] = {
            'interpolation': partner_list,
            'duration': duration_list,
            'true_duration_list': true_duration_list,
            'rois': roi_dict[partner_key]}

    else: 
        target_list = []
        duration_list = []
        true_duration_list = []
        
        for in_epoch in range(0, len(target)):
            target_epoch = target[in_epoch]
            target_interp = target_epoch.get_data(copy = False, verbose=verbosity)[0]
            duration_list.append(np.shape(target_interp)[1] / current_freq)
            # Compare the duration with the true duration
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
                target_interp = target_interp[:,:round(task_duration_secs*current_freq)]
            else:
                true_duration = np.nan
            true_duration_list.append(true_duration)
            durations.append([targetID, session_n, in_epoch, duration_list[in_epoch], true_duration])

            target_list.append(target_interp)

        # update dictionaries
        data_dict[key] = {
            'interpolation': target_list,
            'duration': duration_list,
            'true_duration': true_duration_list,
            'rois': roi_dict[key]}
        
        error_log.append((f'sub-{targetID}_session-{session_n}', 'partner data missing'))

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
ts_one_brain_session, ts_two_blocks_session, ts_four_blocks_session = [], [], []
doc_one_brain, doc_two_blocks, doc_four_blocks = [], [], []
doc_one_brain_session, doc_two_blocks_session, doc_four_blocks_session = [], [], []
rois_one_brain, rois_two_blocks, rois_four_blocks = [], [], []
rois_one_brain_session, rois_two_blocks_session, rois_four_blocks_session = [], [], []
key_list = set(data_dict.keys())
for i, row in dyads.iterrows():
    for session in range(6):
        target_key = f"sub-{row['pID1']}_session-{session + 1}"
        
        if target_key in key_list: 
            target_list = data_dict[target_key]['interpolation']
            target_rois = data_dict[target_key]['rois']
            partner_key = f"sub-{row['pID2']}_session-{session + 1}"
            ts_target_temp, ts_partner_temp, ts_two_temp, ts_four_temp = [], [], [], []
            for activity in range(4):
                target_ts = target_list[activity]
                # channel-wise z-scoring
                target_ts = sp.stats.zscore(target_ts, axis=1, ddof=1)
                # 1a target, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                ts_one_brain.append(target_ts)
                doc_one_brain.extend([[row['pID1'], session, activity]])
                rois_one_brain.append(target_rois)
                ts_target_temp.append(target_ts)
            
                if partner_key in key_list: 
                    
                    partner_list = data_dict[partner_key]['interpolation']
                    partner_rois = data_dict[partner_key]['rois']
                    partner_ts = partner_list[activity]
                    # channel-wise z-scoring
                    partner_ts = sp.stats.zscore(partner_ts, axis=1, ddof=1)
                    # 1a partner, one brain data (two blocks: HbO + HbR), channel-wise z-scored
                    ts_one_brain.append(partner_ts)
                    doc_one_brain.extend([[row['pID2'], session, activity]])
                    rois_one_brain.append(partner_rois)
                    ts_partner_temp.append(partner_ts)
                    
                    # 2a, two blocks: target + partner, channel-wise z-scored
                    ts_two = np.concatenate((target_ts, partner_ts), axis = 0)         
                    ts_two_blocks.append(ts_two)
                    doc_two_blocks.extend([[row['dyadID'], session, activity]])
                    rois_two_blocks.append(target_rois + partner_rois)
                    ts_two_temp.append(ts_two)
                    
                    # 3a, four blocks: target HbO, partner HbO, target HbR, partner HbR
                    ts_four = np.concatenate((target_ts[:int(len(target_rois)/2)], partner_ts[:int(len(partner_rois)/2)], target_ts[int(len(target_rois)/2):len(target_rois)], partner_ts[int(len(partner_rois)/2):len(partner_rois)]), axis=0)
                    ts_four_blocks.append(ts_four)
                    doc_four_blocks.extend([[row['dyadID'], session, activity]])
                    rois_four_blocks.append(target_rois + partner_rois)
                    ts_four_temp.append(ts_four)
            # 1b target, one brain data as above, channel- and session-wise z-scored
            ## session-wise z-scoring
            n_samples = [ts.shape[1] for ts in ts_target_temp]
            ts_target_temp = sp.stats.zscore(np.concatenate(ts_target_temp, axis = 1), ddof=1)
            ts_target_split = np.split(ts_target_temp, np.cumsum(n_samples)[:-1], axis=1)

            ## append to list
            for activity in range(4):
                ts_one_brain_session.append(ts_target_split[activity])
                doc_one_brain_session.extend([[row['pID1'], session, activity]])
                rois_one_brain_session.append(target_rois)
            if partner_key in key_list: 
                ## session-wise z-scoring
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
                    # 1b partner, one brain data as above, channel- and session-wise z-scored
                    ts_one_brain_session.append(ts_partner_split[activity])
                    doc_one_brain_session.extend([[row['pID2'], session, activity]])
                    rois_one_brain_session.append(partner_rois)
                    # 2b, two blocks as above, channel- and session-wise z-scored
                    ts_two_blocks_session.append(ts_two_split[activity])
                    doc_two_blocks_session.extend([[row['dyadID'], session, activity]])
                    rois_two_blocks_session.append(target_rois + partner_rois)
                    # 3b, four blocks as above, channel- and session-wise z-scored
                    ts_four_blocks_session.append(ts_four_split[activity])
                    doc_four_blocks_session.extend([[row['dyadID'], session, activity]])
                    rois_four_blocks_session.append(target_rois + partner_rois)

doc_one_brain = pd.DataFrame(doc_one_brain)
doc_two_blocks = pd.DataFrame(doc_two_blocks)
doc_four_blocks = pd.DataFrame(doc_four_blocks)
doc_one_brain_session = pd.DataFrame(doc_one_brain_session)
doc_two_blocks_session = pd.DataFrame(doc_two_blocks_session)
doc_four_blocks_session = pd.DataFrame(doc_four_blocks_session)

# save
doc_one_brain.to_csv(str(path + 'doc_one_brain.csv'))
np.savez(str(path + 'ts_one_brain'), *ts_one_brain)
np.savez(str(path + 'rois_one_brain'), *rois_one_brain)
doc_two_blocks.to_csv(str(path + 'doc_two_blocks.csv'))
np.savez(str(path + 'ts_two_blocks'), *ts_two_blocks)
np.savez(str(path + 'rois_two_blocks'), *rois_two_blocks)
doc_four_blocks.to_csv(str(path + 'doc_four_blocks.csv'))
np.savez(str(path + 'ts_four_blocks'), *ts_four_blocks)
np.savez(str(path + 'rois_four_blocks'), *rois_four_blocks)
doc_one_brain_session.to_csv(str(path + 'doc_one_brain_session.csv'))
np.savez(str(path + 'ts_one_brain_session'), *ts_one_brain_session)
np.savez(str(path + 'rois_one_brain_session'), *rois_one_brain_session)
doc_two_blocks_session.to_csv(str(path + 'doc_two_blocks_session.csv'))
np.savez(str(path + 'ts_two_blocks_session'), *ts_two_blocks_session)
np.savez(str(path + 'rois_two_blocks_session'), *rois_two_blocks_session)
doc_four_blocks_session.to_csv(str(path + 'doc_four_blocks_session.csv'))
np.savez(str(path + 'ts_four_blocks_session'), *ts_four_blocks_session)
np.savez(str(path + 'rois_four_blocks_session'), *rois_four_blocks_session)