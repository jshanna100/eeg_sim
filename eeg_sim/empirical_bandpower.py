import mne
from os.path import isdir
from os import listdir
from utils import (build_band_samples, plot_samples, band_multivar_gauss_est,
                   plot_covar_mats)
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

filenames = listdir(eeg_dir)
raws = []
for filename in filenames:
    if "-raw.fif" not in filename:
        continue
    raws.append(mne.io.Raw(eeg_dir+filename))

samples = build_band_samples(raws, bands, n_jobs=n_jobs, n_fft=500, log=log)
band_distros = band_multivar_gauss_est(samples)
plot_samples(samples, bands, samples["ch_names"])
plot_covar_mats(band_distros)

with open("{}eeg_sim/mats/empirical_distro.pickle".format(base_dir), "wb") as f:
    pickle.dump(band_distros, f)
