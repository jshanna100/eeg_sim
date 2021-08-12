from scipy.signal import cwt, hilbert
import numpy as np
import mne
import pywt
from utils import sacc_wavelet
import matplotlib.pyplot as plt
from os.path import isdir
from os import listdir
plt.ion()

wavelet_scales = (0.1, 0.2) # 40-80Hz at sampling of 250Hz

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/light/"

filenames = listdir(eeg_dir)
epos = []
rates = []
for filename in filenames:
    if "-raw.fif" not in filename:
        continue
    raw = mne.io.Raw(eeg_dir+filename, preload=True)
    raw_eog = raw.copy().pick_types(eog=True)
    eog_data = raw.get_data().mean(axis=0, keepdims=True)
    pz = raw.copy().pick_channels(["Pz"]).get_data()
    reog_data = eog_data - pz
    info = mne.create_info(["reog"], raw.info["sfreq"], ch_types="eog")
    raw_a = mne.io.RawArray(reog_data, info)
    raw.add_channels([raw_a], force_update_info=True)
    coef, freqs = pywt.cwt(reog_data * -1, np.linspace(*wavelet_scales, 20), "gaus1",
                           sampling_period=0.025)
    info = mne.create_info(["sacc_wav"], raw.info["sfreq"])
    coef = coef.mean(axis=0)
    raw_a = mne.io.RawArray(coef, info)
    raw.add_channels([raw_a], force_update_info=True)

    thresh = (np.quantile(coef[0,], 0.99) -
              np.quantile(coef[0,], 0.01))
    peaks, mags = mne.preprocessing.peak_finder(coef[0,], thresh=thresh)
    for peak in peaks:
        raw.annotations.append(raw.times[peak], 0, "Micro-saccade")

    if len(peaks) < 2000:
        print("Fewer than 2000 micro-saccades; data likely faulty. Skipping.")
        continue

    events = mne.events_from_annotations(raw, event_id={"Micro-saccade":1})[0]
    epo = mne.Epochs(raw, events, tmin=-0.25, tmax=0.25, baseline=(None, None),
                     reject={"eeg":250e-6})
    epos.append(epo)
    rates.append(len(peaks)/raw.times[-1])

    ## viz of individual sessions
    # evo = epo.average(picks="all")
    # evo.crop(tmin=-0.05, tmax=0.05)
    # reog_data = evo.copy().pick_channels(["reog"]).data
    # min_time = evo.times[reog_data[0,].argmin()]
    # max_time = evo.times[reog_data[0,].argmax()]
    # evo.plot_joint(times=[min_time, max_time])
