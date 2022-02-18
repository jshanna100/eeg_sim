import numpy as np
import mne
from scipy.linalg import eigh
from os.path import isdir
import matplotlib.pyplot as plt
from sklearn.covariance import ShrunkCovariance
from mne.viz import plot_topomap
import re
plt.ion()

class SPoC_Noise(mne.decoding.SPoC):
    def __init__(self, n_components=4, reg=None, log=None,
                 cov_method_params=None, rank=None):
        super().__init__(n_components=n_components, reg=reg, log=log,
                         transform_into="csp_space",
                         cov_method_params=cov_method_params, rank=rank)

    def fit(self, inst, target, picks=None):
        if isinstance(inst, mne.io.BaseRaw):
            data = inst.get_data()
        elif isinstance(inst, mne.BaseEpochs):
            data = inst.get_data()
            data = np.transpose(data, [1, 0, 2])
            data = data.reshape((len(data), -1))
        if picks is None:
            picks = mne.pick_types(inst.inst, eeg=True)
        pass

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

    def fit(self, data_a, data_b):
        # scale from volts to microvolts for numerical stability
        data_a, data_b = data_a * 1e+6, data_b * 1e+6
        cov_a = ShrunkCovariance().fit(data_a.T).covariance_
        cov_b = ShrunkCovariance().fit(data_b.T).covariance_
        eig_vals, eig_vecs = eigh(cov_a, cov_b)
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
            ax.set_title("{}, {:.3f}".format(comp_idx, self.eig_vals[comp_idx]))
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

raw = mne.io.Raw("{}089_2_marked-raw.fif".format(eeg_dir))
events, _ = mne.events_from_annotations(raw)
epo = mne.Epochs(raw, events, tmin=-0.01, tmax=0.01, baseline=None,
                 event_repeated="drop")
data_a = epo.get_data(picks="eeg")
data_a = np.transpose(data_a, [1, 0, 2])
data_a = np.reshape(data_a, (len(data_a), -1))

data_b = raw.get_data(picks="eeg")

sn = SASS_Noise()
sn.fit(data_a, data_b)
sn.get_sources(raw)
sn.plot_topos(raw.info)
