from os import listdir
from os.path import isdir
import mne
import numpy as np
from anoar import BadChannelFind
from mne.preprocessing import annotate_muscle_zscore, find_eog_events

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"

# this has two modes. if do_reog, then prepare the data with an REOG channel
# and without EOG/EMG artefact marking - this is for usage with wavelet_filter.py
# if not do_reog, then create a normal EOG channel, and identify artefact segments.
# this is for use with empirical_bandpower.py
do_reog = True

raw_dir = base_dir + "hdd/memtacs/raw/"
if do_reog:
    proc_dir = base_dir + "hdd/memtacs/proc/reog/"
else:
    proc_dir = base_dir + "hdd/memtacs/proc/"


l_freq, h_freq = 0.3, 100
overwrite = True

filelist = listdir(raw_dir)
proclist = listdir(proc_dir)
for filename in filelist:
    if isdir(raw_dir+filename): # subjects are organised by dir here
        subj_name = filename
        this_path = raw_dir+filename+"/EEG/"
        this_filelist = listdir(this_path)
        set_idx = 0
        for this_filename in this_filelist: # get every recording in this dir
            if ".vhdr" not in this_filename:
                continue
            try:
                raw = mne.io.read_raw_brainvision(this_path+this_filename,
                                                  preload=True)
            except:
                continue

            # # set the channel types, so MNE knows what they are
            # chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog"}
            # orig_chans = list(chan_dict.keys())
            # for k,v in chan_dict.items():
            #     raw.set_channel_types({k:v})

            # filter
            raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw.notch_filter(np.arange(50, 500, 50), picks="all")

            # reference the EOG channels against each other, add that as
            # eog_v, eog_h
            data = np.empty((0,len(raw)))
            eog_v_picks = mne.pick_channels(raw.ch_names, include=["Vo", "Vu"])
            temp_data = raw.get_data()[eog_v_picks,]
            data = np.vstack((data, temp_data[0,] - temp_data[1,]))
            eog_h_picks = mne.pick_channels(raw.ch_names, include=["Re", "Li"])
            temp_data = raw.get_data()[eog_h_picks,]
            data = np.vstack((data, temp_data[0,] - temp_data[1,]))
            info = mne.create_info(["eog_v","eog_h"], sfreq=raw.info["sfreq"],
                                   ch_types=["eog","eog"])
            non_eeg = mne.io.RawArray(data, info)
            raw.add_channels([non_eeg], force_update_info=True)

            # set default channels locations
            #raw.set_montage("standard_1005")
            # detect bad channels
            picks = mne.pick_types(raw.info, eeg=True)
            bcf = BadChannelFind(picks, thresh=0.8)
            bad_chans = bcf.recommend(raw)
            print(bad_chans)
            raw.info["bads"].extend(bad_chans)

            if not do_reog:
                ## identify artefacts
                # EOG
                eog_events = find_eog_events(raw)
                onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
                durations = [0.5] * len(eog_events)
                descriptions = ['bad blink'] * len(eog_events)
                blink_annot = mne.Annotations(onsets, durations, descriptions)
                # muscle spasms
                muscle_annot, muscle_scores = annotate_muscle_zscore(raw)

                annots = blink_annot + muscle_annot
                raw.set_annotations(annots)

            # interpolate bad channels
            raw.interpolate_bads()

            # downsample to 250Hz
            #raw.resample(250, n_jobs=4)

            # save
            raw.save("{}{}_{}-raw.fif".format(proc_dir, subj_name, set_idx),
                     overwrite=overwrite)
            set_idx += 1
