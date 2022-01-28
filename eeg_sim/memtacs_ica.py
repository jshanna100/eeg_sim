from os import listdir
from os.path import isdir
import mne
import numpy as np
import re
from mne.preprocessing import ICA
from mne.time_frequency import psd_multitaper
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

def band_means(psd, freqs, bands):
    means = {}
    for band_k, band_v in bands.items():
        inds = (freqs >= band_v[0]) & ( freqs < band_v[1])
        means[band_k] = np.mean(psd[:, inds], axis=1)
    return means

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
proc_dir = base_dir + "hdd/memtacs/proc/reog/"

bands = {"delta":[1,4], "theta":[4,7], "alpha":[8,12], "beta":[13,31]}
ch_names = ['Fp1', 'AFz', 'Fp2', 'F7', 'F3', 'F4', 'Fz', 'F8', 'FC5', 'FC1',
            'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
            'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

proclist = listdir(proc_dir)
var_list = ch_names + list(bands.keys()) + ["Subj", "VarExpl"]
df_dict = {var:[] for var in var_list}
for filename in proclist:
    if not re.match("MT-.*-raw.fif", filename):
        continue
    raw = mne.io.Raw(proc_dir + filename, preload=True)
    raw.filter(l_freq=1, h_freq=30)

    # get rid of obvious EOG components
    ica = ICA(0.99)
    ica.fit(raw)
    comps, scores = ica.find_bads_eog(raw, measure="correlation", threshold=0.8)
    raw_noeog = ica.apply(raw, exclude=comps)

    # now mark off all the EOG channels so they're excluded
    chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog"}
    raw_noeog.set_channel_types(chan_dict)

    # brain ica
    ica_noeog = ICA(0.99)
    ica_noeog.fit(raw_noeog)

    # get components
    comps = np.dot(ica_noeog.mixing_matrix_.T,
                   ica_noeog.pca_components_[:ica_noeog.n_components_])

    # spectral analysis for component sources
    srcs = ica_noeog.get_sources(raw_noeog)
    psd, freqs = psd_multitaper(srcs, picks=srcs.ch_names, fmin=1, fmax=30)
    b_means = np.array(list(band_means(psd, freqs, bands).values())).T
    breg = 1 / b_means.sum(axis=1)
    b_means = np.dot(np.diag(breg), b_means)

    # ratio of explained variance
    ratios = ica.pca_explained_variance_ / sum(ica.pca_explained_variance_)
    ratios = ratios[:ica_noeog.n_components_]

    for comp_idx in range(ica_noeog.n_components_):
        for ch_idx, ch in enumerate(ica_noeog.ch_names):
            df_dict[ch].append(comps[comp_idx, ch_idx])
        for band_idx, band in enumerate(list(bands.keys())):
            df_dict[band].append(b_means[comp_idx, band_idx])
        df_dict["Subj"].append(filename)
        df_dict["VarExpl"].append(ratios[comp_idx])

df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}comp_vecs.pickle".format(proc_dir))
