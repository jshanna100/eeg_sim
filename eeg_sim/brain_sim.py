import numpy as np
import mne
from mne.preprocessing import read_ica
import pandas as pd
from os.path import isdir

def sim_series(sim_n, df_counts, df, eeg_chans, sources):
    raws = []
    for sim_idx in range(sim_n):
        raw = None
        while raw is None:
            raw = simulate_brain_raw(df_counts, df, eeg_chans, sources,
                                     sfreq=1000, dur=30., itx_lim=25000)
        raws.append(raw)
    raws[0].append(raws[1:])
    new_raw = raws[0]
    data = new_raw.get_data()
    for annot in new_raw.annotations:
        time_idx = new_raw.time_as_index(annot["onset"])[0]
        for chan_idx in range(len(data)):
            conv = np.convolve(data[chan_idx, time_idx-5:time_idx+5],
                               np.ones(5)/5, mode="same")
            data[chan_idx, time_idx-5:time_idx+5] = conv
    new_raw._data = data
    new_raw.annotations.delete(np.arange(len(new_raw.annotations)))

    return new_raw

def simulate_brain_raw(df_counts, df, eeg_chans, sources, sfreq=1000, dur=30.,
                       itx_lim=10000):
    clusts = df["Cluster"].values
    clusts_unq = np.unique(clusts)
    clust_inds = {x:np.where(clusts==x)[0] for x in clusts_unq}

    counts = dict(df_counts.iloc[np.random.randint(len(df_counts))])

    for itx in range(itx_lim):
        comp_inds = []
        for k,v in counts.items():
            if k == -1: # no eog clusters
                continue
            for vv in np.arange(v):
                rand_comp_idx = np.random.choice(clust_inds[k])
                comp_inds.append(rand_comp_idx)
        total_var = df.iloc[comp_inds,]["VarExpl"].sum()
    if total_var < 0.15 or total_var > 1.:
        print("Could not find suitable combination of components after "
              "{} iterations.".format(itx_lim))
        return None
    print("\n\nFound component combination that explains {:.3f} variance "
          "after {} tries.\n\n".format(total_var, itx))

    raw_arr = np.zeros((len(eeg_chans), int(sfreq*dur)))
    for comp_idx in comp_inds:
        df_row = df.iloc[comp_idx,]
        loc_idx = df_row["LocIdx"]
        src_idx = df_row["SrcIdx"]
        ica = read_ica("{}{}".format(proc_dir, df_row["ICAFile"]))
        src = sources[src_idx][loc_idx,][:, np.newaxis]
        to_pca = np.dot(ica.mixing_matrix_[loc_idx,],
                        ica.pca_components_[:ica.n_components_])[np.newaxis,]
        data = np.dot(src, to_pca)
        data += ica.pca_mean_
        data = np.dot(data, np.diag(ica.pre_whitener_[:,0]))
        # reorder channels in case they're different across ICAs
        chan_inds = np.array([ica.ch_names.index(ch) for ch in eeg_chans])

        raw_arr += data.T[:, :raw_arr.shape[-1]]

    info = mne.create_info(eeg_chans, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(raw_arr, info)

    return raw


if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
proc_dir = base_dir + "hdd/memtacs/proc/reog/"

df = pd.read_pickle("{}comp_vecs_clust.pickle".format(proc_dir))
df_counts = pd.read_pickle("{}comp_vecs_clust_counts.pickle".format(proc_dir))

df_chs = list(df.columns[8:-7]) # eeg channels from df

# get component sources
print("Loading component sources.")
sources = np.load("{}comp_sources.npy".format(proc_dir), allow_pickle=True)

# gamma

df_gamma = pd.read_pickle("{}comp_vecs_clust_gamma.pickle".format(proc_dir))
df_counts_gamma = pd.read_pickle("{}comp_vecs_clust_counts_gamma.pickle".format(proc_dir))

df_chs_gamma = list(df_gamma.columns[8:-7]) # eeg channels from df

for raw_idx in range(10):

    raw = sim_series(12, df_counts, df, df_chs, sources)
    raw_gamma = sim_series(12, df_counts_gamma, df_gamma, df_chs_gamma,
                           sources)

    raw._data += raw_gamma._data

    raw.save("{}sim_brain_{}-raw.fif".format(proc_dir, raw_idx))
