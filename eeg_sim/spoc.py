import numpy as np
import mne
from scipy.linalg import eigh
from os.path import isdir
from os import listdir
import matplotlib.pyplot as plt
from sklearn.covariance import ShrunkCovariance, EmpiricalCovariance
from mne.viz import plot_topomap
from mne.decoding import SPoC
from mne.preprocessing.ica import _find_sources
from mne.preprocessing import ICA
from mne.preprocessing.bads import _find_outliers
import re
plt.ion()

class SPoC_Noise(mne.decoding.SPoC):
    def __init__(self, n_components=4, reg=None, log=None,
                 cov_method_params=None, rank=None):
        super().__init__(n_components=n_components, reg=reg, log=log,
                         transform_into="csp_space",
                         cov_method_params=cov_method_params, rank=rank)
        self.bad_inds = None

    def remix(self, inst, exclude=None):
        exclude = self.bad_inds if exclude is None else exclude
        patterns = self.patterns_
        if isinstance(inst, mne.Epochs):
            epo_data = epo.get_data(picks="eeg")
            trans = np.hstack(self.transform(epo_data))
            if exclude is not None:
                for excl in exclude:
                    trans[:, excl, :] = np.zeros_like(trans[:, excl, :])
            data = np.asarray([np.dot(patterns.T, epoch) for epoch in src])
            new_inst = mne.EpochsArray(data, inst.info)
        elif isinstance(inst, mne.io.Raw):
            trans = np.dot(self.filters_, inst._data)
            if exclude is not None:
                for excl in exclude:
                    trans[excl, :] = np.zeros_like(trans[excl, :])
            data = np.dot(patterns.T, trans)
            new_inst = mne.io.RawArray(data, inst.info)
        else:
            raise ValueError("Inst must be Raw or Epoch.")

        return new_inst

    def find_bads(self, inst, chan, threshold=None, measure="zscore"):
        if isinstance(inst, mne.Epochs):
            epo_data = inst.get_data(picks="eeg")
            trans = np.hstack(self.transform(epo_data))
            chan_data = np.hstack(epo.get_data([chan]))
        elif isinstance(inst, mne.io.Raw):
            raw_data = inst.get_data(picks="eeg")
            trans = np.dot(self.filters_, raw_data)
            chan_data = inst.get_data(picks=chan)
        else:
            raise ValueError("Inst must be Raw or Epoch.")
        scores = _find_sources(trans, chan_data, "pearsonr")
        if measure == "zscore":
            threshold = 3 if threshold is None else threshold
            this_idx = _find_outliers(scores, threshold=threshold)
        elif measure == "correlation":
            threshold = 0.33 if threshold is None else threshold
            this_idx = np.where(abs(scores) > threshold)[0]
        print(this_idx)
        self.bad_inds = this_idx

class SASS_Noise():
    def __init__(self):
        self.src = None

    def onpick(self, event):
        if self.src is None:
            print("Sources not yet calculated; displaying nothing.")
            return

        # hack
        string = event.artist.title.get_text()
        idx = int(re.match("(\d*),.*", string).groups(1)[0])

        mne_fig = self.src.plot_psd(picks=idx, fmax=100)
        mne_fig.axes[0].set_title(idx)

    def fit(self, data_a, data_b, estimator):
        # scale from volts to microvolts for numerical stability
        data_a, data_b = data_a * 1e+6, data_b * 1e+6
        cov_a = estimator.fit(data_a.T).covariance_
        cov_b = estimator.fit(data_b.T).covariance_
        eig_vals, eig_vecs = eigh(cov_a, cov_a+cov_b)
        # reverse the order
        rev_inds = np.arange(len(eig_vals)-1, -1, -1)
        eig_vals = eig_vals[rev_inds]
        eig_vecs = eig_vecs[:, rev_inds]

        self.eig_vals = eig_vals.T
        self.unmix = eig_vecs.T
        self.mix = np.linalg.pinv(self.unmix)

    def plot_topos(self, info, row_n=4, col_n=8):
        fig, axes = plt.subplots(row_n, col_n)
        axes = [ax for axe in axes for ax in axe]
        for comp_idx, ax in zip(range(len(self.eig_vals)), axes):
            plot_topomap(self.unmix[comp_idx,], info, axes=ax)
            ax.set_title("{}, {:.2f}".format(comp_idx, self.eig_vals[comp_idx]))
            ax.set_picker(True)
        fig.canvas.mpl_connect("pick_event", self.onpick)

    def get_sources(self, raw):
        new_info = mne.create_info(len(self.eig_vals), raw.info["sfreq"])
        src_data = np.dot(self.unmix, raw.get_data(picks="eeg") * 1e+6)
        src = mne.io.RawArray(src_data, new_info)
        self.src = src

        return src

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

threshold = 2

filelist = listdir(eeg_dir)
bad_comps = []
for filename in filelist:
    match = re.match("(\d+._\d)_marked-raw.fif", filename)
    if not match:
        continue

    file_id = match.groups(1)[0]
    raw = mne.io.Raw("{}{}".format(eeg_dir, filename),
                     preload=True)
    raw.filter(l_freq=30, h_freq=80)
    raw.filter(l_freq=30, h_freq=80, picks="reog")
    raw.notch_filter(50)
    raw.notch_filter(50, picks="reog")

    epo = mne.make_fixed_length_epochs(raw, duration=0.05, preload=True)
    sacc_wav = epo.copy().pick_channels(["sacc_wav"]).get_data()
    sacc_max = sacc_wav.max(axis=-1).squeeze()

    spoc = SPoC_Noise(n_components=32, reg=0.5)
    epo_data = epo.get_data(picks="eeg")
    spoc.fit(epo_data, sacc_max)

    raw_eeg = raw.copy().pick_types(eeg=True)
    spoc.find_bads(raw, "reog", threshold=threshold)
    bad_comps.append(spoc.bad_inds)
    spoc_raw = spoc.remix(raw_eeg)
    spoc_raw.save("{}{}_spoc_marked-raw.fif".format(eeg_dir, file_id),
                  overwrite=True)

    ica = ICA(method="picard")
    keep_chan_inds = np.hstack((mne.pick_types(raw.info, eeg=True),
                               mne.pick_channels(raw.ch_names, ["reog"])))
    chans = [raw.ch_names[idx] for idx in keep_chan_inds]
    raw_ica = raw.pick_channels(chans)
    ica.fit(raw_ica)
    bads, _ = ica.find_bads_eog(raw_ica, threshold=threshold)
    ica_raw = ica.apply(raw_ica, exclude=bads)
    ica_raw.save("{}{}_ica_marked-raw.fif".format(eeg_dir, file_id),
                 overwrite=True)
