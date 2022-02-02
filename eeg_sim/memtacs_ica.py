from os import listdir
from os.path import isdir
import mne
import numpy as np
import re
from mne.preprocessing import ICA
from mne.time_frequency import psd_multitaper
import matplotlib.pyplot as plt
import pandas as pd
import pickle
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

bands = {"delta":[1,4], "theta":[4,7], "alpha":[8,12], "beta":[13,31],
         "low_gamma":[30, 45], "high_gamma":[55, 85]}
ch_names = ['Fp1', 'AFz', 'Fp2', 'F7', 'F3', 'F4', 'Fz', 'F8', 'FC5', 'FC1',
            'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
            'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

# resting state only
with open("{}resting_state_files.pickle".format(proc_dir), "rb") as f:
    rests = pickle.load(f)

proclist = listdir(proc_dir)
meta_vars = ["Subj", "Chunk", "VarExpl", "Gamma", "CompIdx", "LocIdx",
             "SrcIdx", "ICAFile"]
var_list = meta_vars + ch_names + list(bands.keys())
df_dict = {var:[] for var in var_list}
all_comp_idx = 0
sources = []
src_idx = 0
for filename in proclist:
    if filename not in rests:
        continue
    raw = mne.io.Raw(proc_dir + filename, preload=True)
    raw_gamma = raw.copy().filter(l_freq=30, h_freq=85)
    raw.filter(l_freq=1, h_freq=30)

    # get rid of obvious EOG components
    ica = ICA(0.99, method="picard")
    ica.fit(raw)
    comps, scores = ica.find_bads_eog(raw, measure="correlation", threshold=0.8)
    raw_noeog = ica.apply(raw, exclude=comps)

    # now mark off all the EOG channels so they're excluded
    chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog"}
    raw_noeog.set_channel_types(chan_dict)
    raw_gamma.set_channel_types(chan_dict)

    # brain ica
    chunks = np.arange(0, raw_noeog.times[-1], 30)
    starts, ends = chunks[:-1], chunks[1:]
    for t_idx, (start, end) in enumerate(zip(starts, ends)):
        for is_gamma, which_raw in zip([False, True], [raw_noeog, raw_gamma]):
            this_raw = which_raw.copy().crop(start, end)
            try:
                this_ica = ICA(0.99, method="picard")
                this_ica.fit(this_raw)
            except:
                this_ica = ICA(0.999, method="picard")
                this_ica.fit(this_raw)

            # get components
            comps = this_ica.get_components()

            # spectral analysis for component sources
            srcs = this_ica.get_sources(this_raw)
            psd, freqs = psd_multitaper(srcs, picks=srcs.ch_names, fmin=1, fmax=85)
            b_means = np.array(list(band_means(psd, freqs, bands).values())).T
            breg = 1 / b_means.sum(axis=1)
            b_means = np.dot(np.diag(breg), b_means)
            sources.append(srcs.get_data().astype(np.float32))

            # ratio of explained variance
            ratios = this_ica.pca_explained_variance_ / sum(this_ica.pca_explained_variance_)
            ratios = ratios[:this_ica.n_components_]

            if is_gamma:
                ica_file = "{}_{}_gamma-ica.fif".format(filename[:-8], t_idx)
            else:
                ica_file = "{}_{}-ica.fif".format(filename[:-8], t_idx)
            this_ica.save(proc_dir + ica_file)

            for loc_idx, comp_idx in enumerate(range(this_ica.n_components_)):
                for ch_idx, ch in enumerate(this_ica.ch_names):
                    df_dict[ch].append(comps[ch_idx, comp_idx])
                for band_idx, band in enumerate(list(bands.keys())):
                    df_dict[band].append(b_means[comp_idx, band_idx])
                df_dict["Subj"].append(filename)
                df_dict["Chunk"].append(t_idx)
                df_dict["VarExpl"].append(ratios[comp_idx])
                df_dict["Gamma"].append(is_gamma)
                df_dict["CompIdx"].append(all_comp_idx)
                df_dict["LocIdx"].append(loc_idx)
                df_dict["SrcIdx"].append(src_idx)
                df_dict["ICAFile"].append(ica_file)
                all_comp_idx += 1
            src_idx += 1

df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}comp_vecs.pickle".format(proc_dir))
sources = np.array(sources, dtype=object)
np.save("{}comp_sources.npy".format(proc_dir), sources)
