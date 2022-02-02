import numpy as np
import mne
import pywt
from utils import sacc_wavelet
import matplotlib.pyplot as plt
from os.path import isdir
import pandas as pd
from os import listdir
import pickle
import re
plt.ion()

'''
Catch and label micro-saccades. This should run on the results of
memtacs_preproc.py with the do_reog variable set to True
'''

wavelet_scales = (5, 2.5) # 40-80Hz at a sampling rate of 1000Hz

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

filenames = listdir(eeg_dir)
epos = []
rates = []
minmaxima = []
freqs = []

# resting state only
with open("{}resting_state_files.pickle".format(eeg_dir), "rb") as f:
    rests = pickle.load(f)

for filename in filenames:
    if filename not in rests:
        continue
    match = re.search("MT-YG-(.*)-raw.fif", filename)
    subj_id = match.groups(1)
    raw = mne.io.Raw(eeg_dir+filename, preload=True)

    # create the REOG channel: EOG average - Pz
    raw_eog = raw.copy().pick_channels(["Re", "Li", "Vo", "Vu"])
    eog_data = raw.get_data().mean(axis=0, keepdims=True)
    pz = raw.copy().pick_channels(["Pz"]).get_data()
    reog_data = eog_data - pz
    info = mne.create_info(["reog"], raw.info["sfreq"], ch_types="eog")
    raw_a = mne.io.RawArray(reog_data, info)
    raw.add_channels([raw_a], force_update_info=True)

    # wavelet transform
    coef, freqs = pywt.cwt(reog_data * -1, np.linspace(*wavelet_scales, 50),
                           "gaus1", sampling_period=0.001)
    # add the results of the wavelet transform for later visualisation
    info = mne.create_info(["sacc_wav"], raw.info["sfreq"])
    coef_a = coef.mean(axis=0)
    raw_a = mne.io.RawArray(coef_a, info)
    raw.add_channels([raw_a], force_update_info=True)

    # calculate the threshold for marking something a saccade or not.
    # here, this is the distance between the 99th and 1th quantile, which
    # seems to work reasonably well
    thresh = (np.quantile(coef_a[0,], 0.99) -
              np.quantile(coef_a[0,], 0.01))
    peaks, mags = mne.preprocessing.peak_finder(coef_a[0,], thresh=thresh)
    # mark the peaks as MNE annotations
    for peak in peaks:
        raw.annotations.append(raw.times[peak], 0, "Micro-saccade")

    if (len(peaks) / raw.times[1]) < 1:
        print("Fewer than 2000 micro-saccades; data likely faulty. Skipping.")
        continue

    # convert raw to epoched file, while also creating a pandas dataframe
    # as metadata, for potentially added convenience later
    events = mne.events_from_annotations(raw, event_id={"Micro-saccade":1})[0]
    subj_ids = {"SubjID":[subj_id for x in range(len(events))]}
    df = pd.DataFrame.from_dict(subj_ids)
    epo = mne.Epochs(raw, events, tmin=-0.25, tmax=0.25, baseline=(-0.25, -0.1),
                     reject={"eeg":250e-6}, metadata=df)
    epo.save("{}{}-epo.fif".format(eeg_dir, subj_id[0]), overwrite=True)
    epos.append(epo)

    # get all the stats for the saccades
    # how often they occur
    rates.append(len(peaks)/raw.times[-1])
    # peaks and troughs
    epo.load_data()
    data = epo.copy().crop(tmin=-0.02, tmax=0.02).apply_baseline((None, None)).get_data(picks="reog")
    minima, maxima = data.min(axis=-1)[:,0], data.max(axis=-1)[:,0]
    minmax = np.stack((minima, maxima)).T

    # # viz of individual sessions
    # evo = epo.average(picks="all")
    # evo.crop(tmin=-0.05, tmax=0.05)
    # reog_data = evo.copy().pick_channels(["reog"]).data
    # min_time = evo.times[reog_data[0,].argmin()]
    # max_time = evo.times[reog_data[0,].argmax()]
    # evo.plot_joint(times=[min_time, max_time])
    # plt.figure()
    # plt.imshow(coef[:,0,peaks])

# combine all epochs from all valid recordings into one large Epoch
grand_epo = mne.concatenate_epochs(epos)
grand_epo.reset_drop_log_selection()
grand_epo.save("{}grand_saccade-epo.fif".format(eeg_dir), overwrite=True)
rates = np.array(rates)
np.save("{}rates.npy".format(eeg_dir), rates)
