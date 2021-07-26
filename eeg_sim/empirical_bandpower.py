import mne
from os.path import isdir
from os import listdir
from utils import calc_bandpower
import numpy as np
import pickle
from scipy.stats import sem, norm
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/"


n_jobs = 1
bands = {"theta":(4,8), "alpha":(8,13), "beta":(13,31), "gamma":(30,100)}
log = True

output = {k:{} for k in bands.keys()}
filenames = listdir(eeg_dir)
for filename in filenames:
    raw = mne.io.Raw(eeg_dir+filename)
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, eeg=True)]
    bandpower = calc_bandpower(raw, ch_names, bands, n_jobs=n_jobs, log=log)
    for band in bands.keys():
        for ch in ch_names:
            if ch not in output[band]:
                output[band][ch] = {"vals":[]}
            ch_idx = bandpower["chan_names"].index(ch)
            output[band][ch]["vals"].append(bandpower[band][:, ch_idx])
for band in bands.keys():
    fig, axes = plt.subplots(4,7, figsize=(38.4, 21.6))
    axes = [a for ax in axes for a in ax]
    plt.suptitle(band)
    for ax, ch in zip(axes, ch_names):
        output[band][ch]["vals"] = np.hstack(output[band][ch]["vals"])
        output[band][ch]["mean"] = output[band][ch]["vals"].mean()
        output[band][ch]["std"] = output[band][ch]["vals"].std()
        ax.set_title(ch)
        this_norm = norm(loc=output[band][ch]["mean"],
                         scale=output[band][ch]["std"])
        x = np.linspace(this_norm.ppf(0.001), this_norm.ppf(0.999), 500)
        y = this_norm.pdf(x)
        ax.hist(output[band][ch]["vals"], density=True, alpha=0.5, bins=30)
        ax.plot(x, y)

with open("{}eeg_sim/mats/empirical_distro.pickle".format(base_dir), "wb") as f:
    pickle.dump(output, f)
